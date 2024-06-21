use std::{
    collections::{BTreeMap, HashMap},
    io::{self, Read as _, Write as _},
};

use log::{log_enabled, Level};

const CACHE_SIZE: usize = 64;

pub struct Machine {
    pc: usize,
    regs: [u32; 8],
    mems: Vec<Block>,
    free: Vec<usize>,
    cache: [Vec<Block>; CACHE_SIZE],
    inst_count: usize,
    counter: Box<[usize]>,
    counters: HashMap<usize, Box<[usize]>>,
    block_id: usize,
    mem_profile: MemProfile,
}

#[derive(Default, Clone)]
struct Block {
    id: usize,
    mem: Box<[u32]>,
}

impl Block {
    fn new(id: usize, mem: Box<[u32]>) -> Self {
        Self { id, mem }
    }
}

#[derive(Default)]
struct MemProfile {
    alloc_count: usize,
    alloc_words: usize,
    alloc_histgram: HashMap<usize, usize>,
    cache_hit: usize,
    cache_miss: usize,
    max_lives: usize,
}

impl MemProfile {
    fn report(&self) {
        log::info!("Allocated blocks:  {}", self.alloc_count);
        log::info!("Allocated words:   {}", self.alloc_words);
        log::info!("Alloc cache hit:   {}", self.cache_hit);
        log::info!("Alloc cache miss:  {}", self.cache_miss);
        log::info!(
            "Alloc cache ratio: {:.2}%",
            self.cache_hit as f64 / (self.cache_hit + self.cache_miss) as f64 * 100.0
        );
        log::info!("Max live blocks:   {}", self.max_lives);
        log::info!("Allocated histogram:");
        for (size, count) in self.alloc_histgram.iter().collect::<BTreeMap<_, _>>() {
            log::info!("  {:6}: {}", size, count);
        }
    }

    fn record_alloc(&mut self, size: usize, lives: usize) {
        if log::log_enabled!(log::Level::Info) {
            self.alloc_count += 1;
            self.alloc_words += size;
            *self.alloc_histgram.entry(size).or_default() += 1;
            self.max_lives = self.max_lives.max(lives);
        }
    }
}

impl Drop for Machine {
    fn drop(&mut self) {
        if log::log_enabled!(log::Level::Info) {
            log::info!("Overall instruction count: {}", self.inst_count);

            log::info!("Hotspot profile:");

            self.counters.insert(self.mems[0].id, self.counter.clone());

            for (id, counter) in &self.counters.iter().collect::<BTreeMap<_, _>>() {
                let mut start = 0;
                let mut cnt = 0;

                for pc in 0..counter.len() {
                    if counter[pc] == cnt {
                        continue;
                    }

                    if cnt >= 1000 {
                        log::info!("{id:06}:{start:08x}-{pc:08x}({:04}): {cnt}", pc - start - 1);
                    }

                    cnt = counter[pc];
                    start = pc;
                }
                log::info!("=====");
            }

            log::info!("Memory profile:");
            self.mem_profile.report();
        }
    }
}

impl Machine {
    pub fn from_bin(bin: &[u8]) -> anyhow::Result<Self> {
        let chunks = bin.chunks_exact(4);
        if !chunks.remainder().is_empty() {
            anyhow::bail!("invalid binary");
        }
        let words = chunks
            .into_iter()
            .map(|w| w.try_into().map(u32::from_be_bytes))
            .collect::<Result<_, _>>()?;

        let block = Block::new(0, words);

        let counter = vec![0; block.mem.len()].into_boxed_slice();

        let mut mems = Vec::with_capacity(1 << 16);
        mems.push(block);

        Ok(Self {
            pc: 0,
            regs: [0; 8],
            mems,
            free: Vec::with_capacity(1 << 16),
            cache: [(); CACHE_SIZE].map(|_| Vec::new()),
            inst_count: 0,
            counter,
            counters: Default::default(),
            block_id: 1,
            mem_profile: Default::default(),
        })
    }

    pub fn run(&mut self) {
        loop {
            let opc = self.mems[0].mem[self.pc];
            self.counter[self.pc] += 1;
            self.pc += 1;
            if log_enabled!(Level::Info) {
                self.inst_count += 1;
            }

            let a = (opc >> 6) & 7;
            let b = (opc >> 3) & 7;
            let c = opc & 7;

            match opc >> 28 {
                // Conditional Move
                0 => {
                    if self.regs[c as usize] != 0 {
                        self.regs[a as usize] = self.regs[b as usize];
                    }
                }
                // Array Index
                1 => {
                    self.regs[a as usize] = self.mems[self.regs[b as usize] as usize].mem
                        [self.regs[c as usize] as usize]
                }
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
                    self.mems[self.regs[a as usize] as usize].mem[self.regs[b as usize] as usize] =
                        self.regs[c as usize];
                }
                // Addition
                3 => {
                    self.regs[a as usize] =
                        self.regs[b as usize].wrapping_add(self.regs[c as usize])
                }
                // Multiplication
                4 => {
                    self.regs[a as usize] =
                        self.regs[b as usize].wrapping_mul(self.regs[c as usize])
                }
                // Division
                5 => self.regs[a as usize] = self.regs[b as usize] / self.regs[c as usize],
                // Not-And
                6 => self.regs[a as usize] = !(self.regs[b as usize] & self.regs[c as usize]),
                // Halt
                7 => break,
                // Allocation
                8 => {
                    let size = self.regs[c as usize] as usize;

                    let buf = if size < self.cache.len() {
                        if let Some(mut block) = self.cache[size].pop() {
                            self.mem_profile.cache_hit += 1;
                            block.id = self.block_id;
                            self.block_id += 1;
                            block.mem.fill(0);
                            block
                        } else {
                            self.mem_profile.cache_miss += 1;
                            let mem = vec![0; size].into_boxed_slice();
                            let id = self.block_id;
                            self.block_id += 1;
                            Block::new(id, mem)
                        }
                    } else {
                        // eprintln!("Allocating a large buffer: {} words", size);
                        self.mem_profile.cache_miss += 1;
                        let mem = vec![0; size].into_boxed_slice();
                        let id = self.block_id;
                        self.block_id += 1;
                        Block::new(id, mem)
                    };

                    if let Some(idx) = self.free.pop() {
                        self.mems[idx] = buf;
                        self.regs[b as usize] = idx as u32;
                    } else {
                        self.regs[b as usize] = self.mems.len() as u32;
                        self.mems.push(buf);
                    }

                    if log_enabled!(Level::Info) {
                        self.mem_profile.record_alloc(size, self.mems.len());
                    }
                }
                // Abandonment
                9 => {
                    let idx = self.regs[c as usize] as usize;
                    let mut buf = Block::default();
                    std::mem::swap(&mut self.mems[idx], &mut buf);
                    let size = buf.mem.len();
                    if size < self.cache.len() {
                        self.cache[size].push(buf);
                    }
                    self.free.push(idx);
                }
                // Output
                10 => {
                    let ch = self.regs[c as usize];
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
                    self.regs[c as usize] = ch;
                }
                // Load Program
                12 => {
                    let idx = self.regs[b as usize] as usize;
                    if idx != 0 {
                        log::info!(
                            "*** LONGJMP to block: {}:{} ***",
                            self.mems[idx].id,
                            self.regs[c as usize]
                        );
                        log::info!("*** Assembly dump:");
                        if log_enabled!(Level::Info) {
                            let code = crate::disasm::disasm(0, &self.mems[idx].mem).unwrap();
                            for instr in code {
                                log::info!("{}", instr);
                            }
                        }

                        let mut progn = self.mems[idx].clone();
                        let mut counter = self
                            .counters
                            .remove_entry(&self.mems[0].id)
                            .map_or_else(|| vec![0; progn.mem.len()].into_boxed_slice(), |r| r.1);

                        std::mem::swap(&mut self.mems[0], &mut progn);
                        std::mem::swap(&mut self.counter, &mut counter);

                        self.counters.insert(progn.id, counter);
                    }
                    self.pc = self.regs[c as usize] as usize;
                }
                // Orthography
                13 => {
                    let a = (opc >> 25) & 7;
                    let val = opc & 0x1ffffff;
                    self.regs[a as usize] = val;
                }
                14 | 15 => panic!("Invalid opcode: {:#x}", opc),
                _ => unreachable!(),
            }
        }
    }
}
