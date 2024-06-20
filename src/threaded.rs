use std::io::{self, Read as _, Write as _};

pub struct Machine {
    pc: usize,
    regs: [u32; 8],
    program: Vec<fn(&mut Machine)>,
    mems: Vec<Box<[u32]>>,
    free: Vec<usize>,
}

fn invalid(_: &mut Machine) {
    panic!("Invalid opcode");
}

fn uncompiled(m: &mut Machine) {
    let f = compile(m.mems[0][m.pc]);
    m.program[m.pc] = f;
    f(m);
}

fn compile(opc: u32) -> fn(&mut Machine) {
    let a = (opc >> 6) & 7;
    let b = (opc >> 3) & 7;
    let c = opc & 7;

    match opc >> 28 {
        0 => cmov::compile(a, b, c),
        1 => array_index::compile(a, b, c),
        2 => array_amendment::compile(a, b, c),
        3 => add::compile(a, b, c),
        4 => mul::compile(a, b, c),
        5 => div::compile(a, b, c),
        6 => nand::compile(a, b, c),
        7 => halt,
        8 => alloc::compile(b, c),
        9 => free::compile(b, c),
        10 => output::compile(c),
        11 => input::compile(c),
        12 => load_program::compile(b, c),
        13 => orthography::compile((opc >> 25) & 7),
        14 | 15 => invalid,
        _ => unreachable!(),
    }
}

macro_rules! gen_compiler {
    ($f:ident) => {
        mod $f {
            use super::Machine;

            pub fn compile(a: u32, b: u32, c: u32) -> fn(&mut Machine) {
                match a {
                    0 => compile2::<0>(b, c),
                    1 => compile2::<1>(b, c),
                    2 => compile2::<2>(b, c),
                    3 => compile2::<3>(b, c),
                    4 => compile2::<4>(b, c),
                    5 => compile2::<5>(b, c),
                    6 => compile2::<6>(b, c),
                    7 => compile2::<7>(b, c),
                    _ => unreachable!(),
                }
            }

            fn compile2<const A: usize>(b: u32, c: u32) -> fn(&mut Machine) {
                match b {
                    0 => compile3::<A, 0>(c),
                    1 => compile3::<A, 1>(c),
                    2 => compile3::<A, 2>(c),
                    3 => compile3::<A, 3>(c),
                    4 => compile3::<A, 4>(c),
                    5 => compile3::<A, 5>(c),
                    6 => compile3::<A, 6>(c),
                    7 => compile3::<A, 7>(c),
                    _ => unreachable!(),
                }
            }

            fn compile3<const A: usize, const B: usize>(c: u32) -> fn(&mut Machine) {
                match c {
                    0 => super::$f::<A, B, 0>,
                    1 => super::$f::<A, B, 1>,
                    2 => super::$f::<A, B, 2>,
                    3 => super::$f::<A, B, 3>,
                    4 => super::$f::<A, B, 4>,
                    5 => super::$f::<A, B, 5>,
                    6 => super::$f::<A, B, 6>,
                    7 => super::$f::<A, B, 7>,
                    _ => unreachable!(),
                }
            }
        }
    };
}

macro_rules! gen_compiler_bc {
    ($f:ident) => {
        mod $f {
            use super::Machine;

            pub fn compile(b: u32, c: u32) -> fn(&mut Machine) {
                match b {
                    0 => compile2::<0>(c),
                    1 => compile2::<1>(c),
                    2 => compile2::<2>(c),
                    3 => compile2::<3>(c),
                    4 => compile2::<4>(c),
                    5 => compile2::<5>(c),
                    6 => compile2::<6>(c),
                    7 => compile2::<7>(c),
                    _ => unreachable!(),
                }
            }

            fn compile2<const B: usize>(c: u32) -> fn(&mut Machine) {
                match c {
                    0 => super::$f::<B, 0>,
                    1 => super::$f::<B, 1>,
                    2 => super::$f::<B, 2>,
                    3 => super::$f::<B, 3>,
                    4 => super::$f::<B, 4>,
                    5 => super::$f::<B, 5>,
                    6 => super::$f::<B, 6>,
                    7 => super::$f::<B, 7>,
                    _ => unreachable!(),
                }
            }
        }
    };
}

macro_rules! gen_compiler_c {
    ($f:ident) => {
        mod $f {
            use super::Machine;

            pub fn compile(c: u32) -> fn(&mut Machine) {
                match c {
                    0 => super::$f::<0>,
                    1 => super::$f::<1>,
                    2 => super::$f::<2>,
                    3 => super::$f::<3>,
                    4 => super::$f::<4>,
                    5 => super::$f::<5>,
                    6 => super::$f::<6>,
                    7 => super::$f::<7>,
                    _ => unreachable!(),
                }
            }
        }
    };
}

fn cmov<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    if m.regs[C] != 0 {
        m.regs[A] = m.regs[B];
    }
}

gen_compiler!(cmov);

fn array_index<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    m.regs[A] = unsafe {
        *m.mems
            .get_unchecked(m.regs[B] as usize)
            .get_unchecked(m.regs[C] as usize)
    }
}

gen_compiler!(array_index);

fn array_amendment<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;

    // if self.regs[a as usize] == 0 {
    //     eprintln!(
    //         "Write to program memory: pc={:#010x}, addr={:#010x}, value={:#010x}->{:#010x}",
    //         self.pc - 1,
    //         self.regs[b as usize],
    //         self.mems[self.regs[a as usize] as usize][self.regs[b as usize] as usize],
    //         self.regs[c as usize],
    //     );
    // }

    let ix = m.regs[A] as usize;
    let addr = m.regs[B] as usize;
    unsafe {
        *m.mems.get_unchecked_mut(ix).get_unchecked_mut(addr) = m.regs[C];
        if ix == 0 {
            *m.program.get_unchecked_mut(addr) = uncompiled;
        }
    }
}

gen_compiler!(array_amendment);

fn add<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    m.regs[A] = m.regs[B].wrapping_add(m.regs[C])
}

gen_compiler!(add);

fn mul<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    m.regs[A] = m.regs[B].wrapping_mul(m.regs[C])
}

gen_compiler!(mul);

fn div<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    m.regs[A] = m.regs[B] / m.regs[C]
}

gen_compiler!(div);

fn nand<const A: usize, const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    m.regs[A] = !(m.regs[B] & m.regs[C])
}

gen_compiler!(nand);

fn halt(_: &mut Machine) {
    std::process::exit(0);
}

fn alloc<const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;
    let size = m.regs[C] as usize;

    if let Some(idx) = m.free.pop() {
        m.mems[idx] = vec![0; size].into_boxed_slice();
        m.regs[B] = idx as u32;
    } else {
        m.mems.push(vec![0; size].into_boxed_slice());
        m.regs[B] = m.mems.len() as u32 - 1;
    }
}

gen_compiler_bc!(alloc);

fn free<const B: usize, const C: usize>(m: &mut Machine) {
    m.pc += 1;

    let idx = m.regs[C] as usize;
    m.mems[idx] = Box::default();
    m.free.push(idx);
}

gen_compiler_bc!(free);

fn output<const C: usize>(m: &mut Machine) {
    m.pc += 1;
    let ch = m.regs[C];
    assert!((0..256).contains(&ch));
    io::stdout().write_all(&[ch as u8]).unwrap();
    io::stdout().flush().unwrap();
}

gen_compiler_c!(output);

fn input<const C: usize>(m: &mut Machine) {
    m.pc += 1;
    let mut buf = [0; 1];
    let ch = io::stdin()
        .read_exact(&mut buf)
        .map_or(!0, |_| buf[0] as u32);
    m.regs[C] = ch;
}

gen_compiler_c!(input);

fn load_program<const B: usize, const C: usize>(m: &mut Machine) {
    let idx = m.regs[B] as usize;
    if idx != 0 {
        m.mems[0] = m.mems[idx].clone();
        m.program = vec![uncompiled; m.mems[0].len()];
    }
    m.pc = m.regs[C] as usize;
}

gen_compiler_bc!(load_program);

fn orthography<const A: usize>(m: &mut Machine) {
    m.regs[A] = unsafe { m.mems[0].get_unchecked(m.pc) } & 0x1ffffff;
    m.pc += 1;
}

gen_compiler_c!(orthography);

impl Machine {
    pub fn from_bin(bin: &[u8]) -> anyhow::Result<Self> {
        let chunks = bin.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            anyhow::bail!("invalid binary");
        }
        let words = chunks
            .into_iter()
            .map(|w| w.try_into().map(u32::from_be_bytes))
            .collect::<Result<Box<[u32]>, _>>()?;

        let ret = Self {
            pc: 0,
            regs: [0; 8],
            program: vec![uncompiled; words.len()],
            mems: vec![words],
            free: vec![],
        };

        Ok(ret)
    }

    pub fn run(&mut self) {
        loop {
            self.program[self.pc](self);
        }
    }
}
