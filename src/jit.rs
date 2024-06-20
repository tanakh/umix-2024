use std::{
    collections::HashMap,
    hash::{Hash as _, Hasher as _},
    io::{self, Read as _, Write as _},
};

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::Module as _;
use log::{log_enabled, Level};

const MAX_BLOCKS: usize = 1 << 20;
const CACHE_SIZE: usize = 32;
const JIT_THRESHOLD: usize = 1024;

type Block = Box<[u32]>;

type CompiledFn = unsafe fn(
    machine: *mut Machine,
    regs: *mut u32,
    mems: *const Block,
    cb: *const Callbacks,
) -> u32;

#[repr(C)]
struct Callbacks {
    putc: extern "C" fn(u32),
    getc: extern "C" fn() -> u32,
    alloc: extern "C" fn(*mut Machine, u32) -> u32,
    free: extern "C" fn(*mut Machine, u32),
    trace: extern "C" fn(u32),
}

pub struct Machine {
    pc: usize,
    regs: [u32; 8],
    mems: Box<[Block]>,
    free: Vec<usize>,
    allocator: Allocator,

    jit: JIT,
    counter: Box<[isize]>,
    compiled: Vec<Option<CompiledFn>>,
    callbacks: Callbacks,
}

struct Allocator {
    cache: [Vec<Block>; CACHE_SIZE],
}

impl Allocator {
    fn new() -> Self {
        Self {
            cache: [(); CACHE_SIZE].map(|_| Vec::new()),
        }
    }

    fn alloc(&mut self, size: usize) -> Block {
        if size < CACHE_SIZE {
            if let Some(mut block) = self.cache[size].pop() {
                block.fill(0);
                block
            } else {
                vec![0; size].into_boxed_slice()
            }
        } else {
            // eprintln!("Allocating a large buffer: {} words", size);
            vec![0; size].into_boxed_slice()
        }
    }

    fn free(&mut self, block: Block) {
        let size = block.len();
        if size > 0 && size < self.cache.len() {
            self.cache[size].push(block);
        }
    }
}

struct JIT {
    module: JITModule,
    ctx: codegen::Context,
    builder_ctx: FunctionBuilderContext,
    cache: HashMap<u64, *const u8>,
    sizes: HashMap<usize, usize>,
}

impl Drop for JIT {
    fn drop(&mut self) {
        if log::log_enabled!(Level::Info) {
            log::info!("*** JIT code sizes:");

            let mut sizes = self.sizes.iter().collect::<Vec<_>>();
            sizes.sort();

            for (size, count) in sizes {
                log::info!("    {} words: {}", size, count);
            }
        }
    }
}

struct Analysis {
    read_reg: [bool; 8],
    write_reg: [bool; 8],
    call_putc: bool,
    call_getc: bool,
    call_alloc: bool,
    call_free: bool,
}

impl Analysis {
    fn new(code: &[u32]) -> Self {
        let mut read_reg = [false; 8];
        let mut write_reg = [false; 8];
        let mut call_putc = false;
        let mut call_getc = false;
        let mut call_alloc = false;
        let mut call_free = false;

        for &opc in code {
            let a = ((opc >> 6) & 7) as usize;
            let b = ((opc >> 3) & 7) as usize;
            let c = (opc & 7) as usize;

            match opc >> 28 {
                // Conditional Move
                0 => {
                    read_reg[a] |= !write_reg[a];
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // Array Index
                1 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // Array Amendment
                2 => {
                    read_reg[a] |= !write_reg[a];
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                }
                // Addition
                3 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // Multiplication
                4 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // Division
                5 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // Not-And
                6 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                    write_reg[a] = true;
                }
                // halt
                7 => {}
                // Allocation
                8 => {
                    read_reg[c] |= !write_reg[c];
                    write_reg[b] = true;
                    call_alloc = true;
                }
                // Abandonment
                9 => {
                    read_reg[c] |= !write_reg[c];
                    call_free = true;
                }
                // Output
                10 => {
                    read_reg[c] |= !write_reg[c];
                    call_putc = true;
                }
                // Input
                11 => {
                    write_reg[c] = true;
                    call_getc = true;
                }
                // Load Program
                12 => {
                    read_reg[b] |= !write_reg[b];
                    read_reg[c] |= !write_reg[c];
                }
                // Orthography
                13 => {
                    let a = ((opc >> 25) & 7) as usize;
                    write_reg[a] = true;
                }

                _ => unreachable!(),
            }
        }

        Self {
            read_reg,
            write_reg,
            call_putc,
            call_getc,
            call_alloc,
            call_free,
        }
    }
}

impl JIT {
    fn new() -> anyhow::Result<Self> {
        let module = JITModule::new(JITBuilder::with_flags(
            &[("opt_level", "speed")],
            cranelift_module::default_libcall_names(),
        )?);
        eprintln!("*** JIT target ISA: {}", module.isa().triple());
        Ok(Self {
            ctx: module.make_context(),
            builder_ctx: FunctionBuilderContext::new(),
            module,
            cache: HashMap::new(),
            sizes: HashMap::new(),
        })
    }

    fn compile(&mut self, pc: u32, code: &[u32]) -> anyhow::Result<*const u8> {
        let hash = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            code.hash(&mut hasher);
            hasher.finish()
        };

        if let Some(&ptr) = self.cache.get(&hash) {
            // eprintln!("*** Hit compile cache");
            return Ok(ptr);
        }

        if log_enabled!(Level::Info) {
            log::info!(
                "*** JIT compiling block at pc={pc:08x}:{:08x} ({} insts)",
                pc + code.len() as u32 - 1,
                code.len()
            );

            for line in crate::disasm::disasm(pc as usize, code)? {
                log::info!("    {line}");
            }
            log::info!("");
        }

        self.code_gen(pc, code);

        if log_enabled!(Level::Info) {
            self.ctx.set_disasm(true);
        }

        let id = self
            .module
            .declare_anonymous_function(&self.ctx.func.signature)?;

        self.module.define_function(id, &mut self.ctx)?;

        if log_enabled!(Level::Info) {
            let code = self.ctx.compiled_code().unwrap().vcode.as_ref().unwrap();
            log::info!("*** Generated Assembly:");
            log::info!("{code}");
        }

        self.module.clear_context(&mut self.ctx);
        self.module.finalize_definitions()?;

        if log_enabled!(Level::Info) {
            *self.sizes.entry(code.len()).or_default() += 1;
        }

        Ok(self.module.get_finalized_function(id))
    }

    fn code_gen(&mut self, pc: u32, code: &[u32]) {
        let anal = Analysis::new(code);

        let pt = self.module.target_config().pointer_type();

        let params = &mut self.ctx.func.signature.params;

        // machine
        params.push(AbiParam::new(pt));
        // regs
        params.push(AbiParam::new(pt));
        // mems
        params.push(AbiParam::new(pt));
        // cb
        params.push(AbiParam::new(pt));

        self.ctx
            .func
            .signature
            .returns
            .push(AbiParam::new(types::I32));

        let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

        let entry_block = builder.create_block();

        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let param_machine = builder.block_params(entry_block)[0];
        let param_regs = builder.block_params(entry_block)[1];
        let param_mems = builder.block_params(entry_block)[2];
        let param_cb = builder.block_params(entry_block)[3];

        let mf = MemFlags::new();

        let putc_sig = if anal.call_putc {
            Some(builder.import_signature(Signature {
                params: vec![AbiParam::new(types::I32)],
                returns: vec![],
                call_conv: isa::CallConv::SystemV,
            }))
        } else {
            None
        };
        let getc_sig = if anal.call_getc {
            Some(builder.import_signature(Signature {
                params: vec![],
                returns: vec![AbiParam::new(types::I32)],
                call_conv: isa::CallConv::SystemV,
            }))
        } else {
            None
        };
        let alloc_sig = if anal.call_alloc {
            Some(builder.import_signature(Signature {
                params: vec![AbiParam::new(pt), AbiParam::new(types::I32)],
                returns: vec![AbiParam::new(types::I32)],
                call_conv: isa::CallConv::SystemV,
            }))
        } else {
            None
        };
        let free_sig = if anal.call_free {
            Some(builder.import_signature(Signature {
                params: vec![AbiParam::new(pt), AbiParam::new(types::I32)],
                returns: vec![],
                call_conv: isa::CallConv::SystemV,
            }))
        } else {
            None
        };

        let pt_size = self.module.target_config().pointer_bytes() as i32;

        let fp_putc = if anal.call_putc {
            Some(builder.ins().load(pt, mf, param_cb, 0))
        } else {
            None
        };
        let fp_getc = if anal.call_getc {
            Some(builder.ins().load(pt, mf, param_cb, pt_size))
        } else {
            None
        };
        let fp_alloc = if anal.call_alloc {
            Some(builder.ins().load(pt, mf, param_cb, pt_size * 2))
        } else {
            None
        };
        let fp_free = if anal.call_free {
            Some(builder.ins().load(pt, mf, param_cb, pt_size * 3))
        } else {
            None
        };

        let mut finished = false;

        #[derive(Clone, Copy)]
        enum Reg {
            Value(Value),
            Const(u32),
        }

        impl Reg {
            fn value(
                &self,
                builder: &mut FunctionBuilder,
                cache: &mut HashMap<u32, Value>,
            ) -> Value {
                match self {
                    Reg::Value(v) => *v,
                    Reg::Const(val) => {
                        if let Some(v) = cache.get(val) {
                            *v
                        } else {
                            let v = builder.ins().iconst(types::I32, *val as i64);
                            cache.insert(*val, v);
                            v
                        }
                    }
                }
            }
        }

        let mut regs = [Reg::Const(0); 8];
        let mut ptr_cache = [None; 8];
        let mut const_cache = HashMap::<u32, Value>::new();

        for i in 0..8 {
            if anal.read_reg[i] {
                regs[i] = Reg::Value(builder.ins().load(types::I32, mf, param_regs, i as i32 * 4));
            }
        }

        macro_rules! finalize {
            () => {
                for i in 0..8 {
                    if anal.write_reg[i] {
                        let v = regs[i].value(&mut builder, &mut const_cache);
                        builder.ins().store(mf, v, param_regs, i as i32 * 4);
                    }
                }
                finished = true;
            };
        }

        for (pc_ofs, opc) in code.iter().cloned().enumerate() {
            assert!(!finished);

            let a = ((opc >> 6) & 7) as usize;
            let b = ((opc >> 3) & 7) as usize;
            let c = (opc & 7) as usize;

            match opc >> 28 {
                // Conditional Move
                0 => {
                    let v = match regs[c] {
                        Reg::Value(rc) => {
                            let ra = regs[a].value(&mut builder, &mut const_cache);
                            let rb = regs[b].value(&mut builder, &mut const_cache);
                            Reg::Value(builder.ins().select(rc, rb, ra))
                        }
                        Reg::Const(vc) => {
                            if vc != 0 {
                                regs[b]
                            } else {
                                regs[a]
                            }
                        }
                    };

                    regs[a] = v;
                    ptr_cache[a] = None;
                }
                // Array Index
                1 => {
                    let mem = if let Some(mem) = ptr_cache[b] {
                        mem
                    } else {
                        let rb = regs[b].value(&mut builder, &mut const_cache);
                        let ix = builder.ins().ishl_imm(rb, 4);
                        let ix = builder.ins().uextend(pt, ix);
                        let addr = builder.ins().iadd(param_mems, ix);
                        builder.ins().load(pt, mf, addr, 0)
                    };
                    ptr_cache[b] = Some(mem);

                    let v = match regs[c] {
                        Reg::Value(rc) => {
                            let ix = builder.ins().ishl_imm(rc, 2);
                            let ix = builder.ins().uextend(pt, ix);
                            let addr = builder.ins().iadd(mem, ix);
                            builder.ins().load(types::I32, mf, addr, 0)
                        }
                        Reg::Const(cc) => builder.ins().load(types::I32, mf, mem, (cc as i32) * 4),
                    };

                    regs[a] = Reg::Value(v);
                    ptr_cache[a] = None;
                }
                // Array Amendment
                2 => {
                    let mem = if let Some(mem) = ptr_cache[a] {
                        mem
                    } else {
                        let ra = regs[a].value(&mut builder, &mut const_cache);
                        let ix = builder.ins().ishl_imm(ra, 4);
                        let ix = builder.ins().uextend(pt, ix);
                        let addr = builder.ins().iadd(param_mems, ix);
                        builder.ins().load(pt, mf, addr, 0)
                    };
                    ptr_cache[a] = Some(mem);

                    let rc = regs[c].value(&mut builder, &mut const_cache);

                    match regs[b] {
                        Reg::Value(rb) => {
                            let ix = builder.ins().ishl_imm(rb, 2);
                            let ix = builder.ins().uextend(pt, ix);
                            let addr = builder.ins().iadd(mem, ix);
                            builder.ins().store(mf, rc, addr, 0);
                        }
                        Reg::Const(cb) => {
                            builder.ins().store(mf, rc, mem, (cb as i32) * 4);
                        }
                    };
                }
                // Addition
                3 => {
                    regs[a] = match (regs[b], regs[c]) {
                        (Reg::Value(vb), Reg::Value(vc)) => Reg::Value(builder.ins().iadd(vb, vc)),
                        (Reg::Value(vb), Reg::Const(cc)) => {
                            Reg::Value(builder.ins().iadd_imm(vb, cc as i64))
                        }
                        (Reg::Const(cb), Reg::Value(vc)) => {
                            Reg::Value(builder.ins().iadd_imm(vc, cb as i64))
                        }
                        (Reg::Const(cb), Reg::Const(cc)) => Reg::Const(cb.wrapping_add(cc)),
                    };
                    ptr_cache[a] = None;
                }
                // Multiplication
                4 => {
                    regs[a] = match (regs[b], regs[c]) {
                        (Reg::Value(vb), Reg::Value(vc)) => Reg::Value(builder.ins().imul(vb, vc)),
                        (Reg::Value(vb), Reg::Const(cc)) => {
                            Reg::Value(builder.ins().imul_imm(vb, cc as i64))
                        }
                        (Reg::Const(cb), Reg::Value(vc)) => {
                            Reg::Value(builder.ins().imul_imm(vc, cb as i64))
                        }
                        (Reg::Const(cb), Reg::Const(cc)) => Reg::Const(cb.wrapping_mul(cc)),
                    };
                    ptr_cache[a] = None;
                }
                // Division
                5 => {
                    regs[a] = match (regs[b], regs[c]) {
                        (Reg::Value(vb), Reg::Value(vc)) => Reg::Value(builder.ins().udiv(vb, vc)),
                        (Reg::Value(vb), Reg::Const(cc)) => {
                            Reg::Value(builder.ins().udiv_imm(vb, cc as i64))
                        }
                        (Reg::Const(cb), Reg::Value(vc)) => {
                            Reg::Value(builder.ins().udiv_imm(vc, cb as i64))
                        }
                        (Reg::Const(cb), Reg::Const(cc)) => Reg::Const(cb / cc),
                    };
                    ptr_cache[a] = None;
                }
                // Not-And
                6 => {
                    let v = if b != c {
                        match (regs[b], regs[c]) {
                            (Reg::Value(vb), Reg::Value(vc)) => {
                                Reg::Value(builder.ins().band(vb, vc))
                            }
                            (Reg::Value(vb), Reg::Const(cc)) => {
                                Reg::Value(builder.ins().band_imm(vb, cc as i64))
                            }
                            (Reg::Const(cb), Reg::Value(vc)) => {
                                Reg::Value(builder.ins().band_imm(vc, cb as i64))
                            }
                            (Reg::Const(cb), Reg::Const(cc)) => Reg::Const(cb & cc),
                        }
                    } else {
                        regs[b]
                    };
                    regs[a] = match v {
                        Reg::Value(v) => Reg::Value(builder.ins().bnot(v)),
                        Reg::Const(val) => Reg::Const(!val),
                    };
                    ptr_cache[a] = None;
                }
                // halt
                7 => {
                    finalize!();
                    let ret = builder.ins().iconst(types::I32, !0);
                    builder.ins().return_(&[ret]);
                }
                // Allocation
                8 => {
                    let rc = regs[c].value(&mut builder, &mut const_cache);
                    let ret = builder.ins().call_indirect(
                        alloc_sig.unwrap(),
                        fp_alloc.unwrap(),
                        &[param_machine, rc],
                    );
                    regs[b] = Reg::Value(builder.inst_results(ret)[0]);
                    ptr_cache.fill(None);
                }
                // Abandonment
                9 => {
                    let rc = regs[c].value(&mut builder, &mut const_cache);
                    builder.ins().call_indirect(
                        free_sig.unwrap(),
                        fp_free.unwrap(),
                        &[param_machine, rc],
                    );
                }
                // Output
                10 => {
                    let vc = regs[c].value(&mut builder, &mut const_cache);
                    builder
                        .ins()
                        .call_indirect(putc_sig.unwrap(), fp_putc.unwrap(), &[vc]);
                }
                // Input
                11 => {
                    let ret = builder
                        .ins()
                        .call_indirect(getc_sig.unwrap(), fp_getc.unwrap(), &[]);
                    regs[c] = Reg::Value(builder.inst_results(ret)[0]);
                    ptr_cache[c] = None;
                }
                // Load Program
                12 => {
                    finalize!();
                    // if B == 0 then return C, otherwise return address of this instruction
                    let cur_pc = builder
                        .ins()
                        .iconst(types::I32, (pc + pc_ofs as u32) as i64);

                    let rb = regs[b].value(&mut builder, &mut const_cache);
                    let rc = regs[c].value(&mut builder, &mut const_cache);

                    let ret_addr = builder.ins().select(rb, cur_pc, rc);
                    builder.ins().return_(&[ret_addr]);
                }
                // Orthography
                13 => {
                    let a = ((opc >> 25) & 7) as usize;
                    let val = (opc & 0x1ffffff) as u32;
                    regs[a] = Reg::Const(val);
                    ptr_cache[a] = None;
                }

                _ => unreachable!(),
            }
        }

        assert!(finished);

        builder.finalize();
    }
}

extern "C" fn putc(ch: u32) {
    assert!((0..256).contains(&ch));
    io::stdout().write_all(&[ch as u8]).unwrap();
    io::stdout().flush().unwrap();
}

extern "C" fn getc() -> u32 {
    let mut buf = [0; 1];
    io::stdin()
        .read_exact(&mut buf)
        .map_or(!0, |_| buf[0] as u32)
}

extern "C" fn alloc(machine: *mut Machine, size: u32) -> u32 {
    unsafe { machine.as_mut().unwrap().alloc(size) }
}

extern "C" fn free(machine: *mut Machine, ix: u32) {
    unsafe { machine.as_mut().unwrap().free(ix) }
}

extern "C" fn trace(pc: u32) {
    eprintln!("*** Trace: pc={pc:08x}");
}

impl Machine {
    pub fn from_bin(bin: &[u8]) -> anyhow::Result<Self> {
        let chunks = bin.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            anyhow::bail!("invalid binary");
        }

        let block: Box<[u32]> = chunks
            .into_iter()
            .map(|w| w.try_into().map(u32::from_be_bytes))
            .collect::<Result<_, _>>()?;

        let counter = vec![0; block.len()].into_boxed_slice();
        let compiled = vec![None; block.len()];

        let mut mems = vec![Default::default(); MAX_BLOCKS].into_boxed_slice();
        mems[0] = block;

        Ok(Self {
            pc: 0,
            regs: [0; 8],
            mems,
            free: (1..MAX_BLOCKS).rev().collect(),
            allocator: Allocator::new(),
            counter,
            jit: JIT::new()?,
            compiled,
            callbacks: Callbacks {
                putc,
                getc,
                alloc,
                free,
                trace,
            },
        })
    }

    pub fn run(&mut self) {
        loop {
            if self.counter[self.pc] >= JIT_THRESHOLD as isize {
                let func = if let Some(func) = self.compiled[self.pc] {
                    Some(func)
                } else if self.can_compile(self.pc) {
                    let func = self.compile_bb(self.pc).unwrap();
                    self.compiled[self.pc] = Some(func);
                    Some(func)
                } else {
                    self.counter[self.pc] = isize::MIN;
                    None
                };

                if let Some(func) = func {
                    // eprintln!("*** JITted func: {func:?}");

                    // eprintln!("*** Calling JIT func:");

                    // eprintln!("    pc = {:08x}", self.pc);
                    // for (rn, reg) in self.regs.iter().enumerate() {
                    //     eprintln!("    r{rn} = {reg:08x}");
                    // }
                    // eprintln!();

                    let next_pc = unsafe {
                        func(
                            self,
                            self.regs.as_mut_ptr(),
                            self.mems.as_ptr(),
                            &self.callbacks,
                        )
                    };

                    // eprintln!("*** Return from JIT fnnc:");

                    // eprintln!("    pc = {:08x}", next_pc);
                    // for (rn, reg) in self.regs.iter().enumerate() {
                    //     eprintln!("    r{rn} = {reg:08x}");
                    // }
                    // eprintln!();

                    self.pc = next_pc as usize;
                    continue;
                }
            }

            self.counter[self.pc] += 1;

            if !self.interpret() {
                break;
            }
        }
    }

    fn interpret(&mut self) -> bool {
        let opc = self.mems[0][self.pc];
        self.pc += 1;

        let a = ((opc >> 6) & 7) as usize;
        let b = ((opc >> 3) & 7) as usize;
        let c = (opc & 7) as usize;

        match opc >> 28 {
            // Conditional Move
            0 => {
                if self.regs[c] != 0 {
                    self.regs[a] = self.regs[b];
                }
            }
            // Array Index
            1 => self.regs[a] = self.mems[self.regs[b] as usize][self.regs[c] as usize],
            // Array Amendment
            2 => {
                // if self.regs[a as usize] == 0 {
                //     eprintln!(
                //         "Write to program memory: pc={:#010x}, addr={:#010x}, value={:#010x}->{:#010x}",
                //         self.pc - 1,
                //         self.regs[b as usize],
                //         self.mems[self.regs[a as usize] as usize][self.regs[b as usize] as usize],
                //         self.regs[c as usize],
                //     );
                // }
                self.mems[self.regs[a] as usize][self.regs[b] as usize] = self.regs[c];
            }
            // Addition
            3 => self.regs[a] = self.regs[b].wrapping_add(self.regs[c]),
            // Multiplication
            4 => self.regs[a] = self.regs[b].wrapping_mul(self.regs[c]),
            // Division
            5 => self.regs[a] = self.regs[b] / self.regs[c],
            // Not-And
            6 => self.regs[a] = !(self.regs[b] & self.regs[c]),
            // Halt
            7 => return false,
            // Allocation
            8 => {
                let size = self.regs[c];
                let ix = self.alloc(size);
                self.regs[b] = ix;
            }
            // Abandonment
            9 => {
                let ix = self.regs[c];
                self.free(ix);
            }
            // Output
            10 => {
                let ch = self.regs[c];
                assert!((0..256).contains(&ch));
                io::stdout().write_all(&[ch as u8]).unwrap();
                io::stdout().flush().unwrap();
            }
            // Input
            11 => {
                let mut buf = [0; 1];
                let ch = io::stdin()
                    .read_exact(&mut buf)
                    .map_or(!0, |_| buf[0] as u32);
                self.regs[c] = ch;
            }
            // Load Program
            12 => {
                let idx = self.regs[b] as usize;
                if idx != 0 {
                    eprintln!("*** LONGJMP to block: {idx} ***");
                    self.mems[0] = self.mems[idx].clone();
                    self.counter = vec![0; self.mems[0].len()].into_boxed_slice();
                    self.compiled = vec![None; self.mems[0].len()];
                }
                self.pc = self.regs[c] as usize;
            }
            // Orthography
            13 => {
                let a = ((opc >> 25) & 7) as usize;
                let val = opc & 0x1ffffff;
                self.regs[a] = val;
            }
            14 | 15 => panic!("Invalid opcode: {:#x}", opc),
            _ => unreachable!(),
        }

        true
    }

    #[inline(never)]
    fn alloc(&mut self, size: u32) -> u32 {
        let ix = self.free.pop().expect("out of memory");
        self.mems[ix] = self.allocator.alloc(size as usize);
        ix as u32
    }

    #[inline(never)]
    fn free(&mut self, ix: u32) {
        let ix = ix as usize;
        let freed = std::mem::replace(&mut self.mems[ix], Block::default());
        self.allocator.free(freed);
        self.free.push(ix);
    }

    fn can_compile(&self, pc: usize) -> bool {
        let opc = self.mems[0][pc];

        match opc >> 28 {
            7 | 12 => return false,
            _ => {}
        }

        true
    }

    fn compile_bb(&mut self, pc: usize) -> anyhow::Result<CompiledFn> {
        let mut code = vec![];
        let mut pc = pc;

        loop {
            let opc = self.mems[0][pc];
            let op = opc >> 28;

            match op {
                7 | 12 => {
                    code.push(opc);
                    break;
                }
                _ => {
                    code.push(opc);
                    pc += 1;
                }
            }
        }

        Ok(unsafe { std::mem::transmute(self.jit.compile(self.pc as u32, &code)?) })
    }
}
