pub fn decode(bin: &[u8]) -> anyhow::Result<Vec<u32>> {
    let chunks = bin.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        anyhow::bail!("invalid binary");
    }
    Ok(chunks
        .into_iter()
        .map(|w| w.try_into().map(u32::from_be_bytes))
        .collect::<Result<_, _>>()?)
}

pub fn disasm(pc_base: usize, progn: &[u32]) -> anyhow::Result<Vec<String>> {
    let mut ret = vec![];

    let mut regs = [None::<u32>; 8];

    for pc_ofs in 0..progn.len() {
        let pc = pc_base + pc_ofs;
        let opc = progn[pc_ofs];

        let a = ((opc >> 6) & 7) as usize;
        let b = ((opc >> 3) & 7) as usize;
        let c = (opc & 7) as usize;

        macro_rules! reg {
            ($rn:expr) => {
                format!(
                    "r{}{}",
                    $rn,
                    if let Some(val) = regs[$rn] {
                        format!("(={:#x})", val)
                    } else {
                        "".to_string()
                    }
                )
            };
        }

        let instr = match opc >> 28 {
            0 => {
                regs[a] = None;
                if let (Some(b), Some(c)) = (regs[b], regs[c]) {
                    if c != 0 {
                        regs[a] = Some(b);
                    }
                }

                if opc == 0 {
                    format!("nop")
                } else {
                    format!("if {} != 0 then r{a} = {}", reg!(c), reg!(b))
                }
            }
            1 => {
                regs[a] = None;
                format!("r{a} = MEM[{}:{}]", reg!(b), reg!(c))
            }
            2 => format!("MEM[{}:{}] = {}", reg!(a), reg!(b), reg!(c)),
            3 => {
                regs[a] = None;
                if let (Some(b), Some(c)) = (regs[b], regs[c]) {
                    regs[a] = Some(b.wrapping_add(c));
                }
                format!("r{a} = {} + {}", reg!(b), reg!(c))
            }
            4 => {
                regs[a] = None;
                if let (Some(b), Some(c)) = (regs[b], regs[c]) {
                    regs[a] = Some(b.wrapping_mul(c));
                }
                format!("r{a} = {} * {}", reg!(b), reg!(c))
            }
            5 => {
                regs[a] = None;
                if let (Some(b), Some(c)) = (regs[b], regs[c]) {
                    regs[a] = Some(b / c);
                }
                format!("r{a} = {} / {}", reg!(b), reg!(c))
            }
            6 => {
                regs[a] = None;
                if let (Some(b), Some(c)) = (regs[b], regs[c]) {
                    regs[a] = Some(!(b & c));
                }

                if b == c {
                    format!("r{a} = !{}", reg!(b))
                } else {
                    format!("r{a} = !({} & {})", reg!(b), reg!(c))
                }
            }
            7 => {
                regs.fill(None);
                "halt".to_string()
            }
            8 => {
                regs[b] = None;
                format!("r{b} = alloc({})", reg!(c))
            }
            9 => format!("free({})", reg!(c)),
            10 => {
                format!(
                    "putc(r{c}{})",
                    if let Some(val) = regs[c] {
                        if val < 256 {
                            format!(" = {:?}", val as u8 as char)
                        } else {
                            format!(" = {:#x}", val)
                        }
                    } else {
                        "".to_string()
                    }
                )
            }
            11 => {
                regs[c] = None;
                format!("r{c} = getc()")
            }
            12 => {
                let ret = if b == 6 {
                    format!("jmp (r6:){}", reg!(c))
                } else {
                    format!("longjmp {}:{}", reg!(b), reg!(c))
                };
                regs.fill(None);
                ret
            }
            13 => {
                let a = (opc >> 25) & 7;
                let val = opc & 0x1ffffff;

                regs[a as usize] = Some(val);

                format!("r{a} = {val:#x}")
            }
            _ => format!("invalid"),
        };

        let malformed = match opc >> 28 {
            0..=6 => {
                let blank = opc & 0b0000_1111_1111_1111_1111_1110_0000_0000;
                blank != 0
            }
            8 | 12 => {
                let blank = opc & 0b0000_1111_1111_1111_1111_1111_1100_0000;
                blank != 0
            }
            9 | 10 | 11 => {
                let blank = opc & 0b0000_1111_1111_1111_1111_1111_1111_1000;
                blank != 0
            }
            7 => {
                let blank = opc & 0b0000_1111_1111_1111_1111_1111_1111_1111;
                blank != 0
            }
            13 => false,
            _ => true,
        };

        if malformed {
            regs.fill(None);
        }

        ret.push(format!(
            "{pc:#010x}: {opc:08x} {} {instr}",
            if malformed { "!" } else { " " }
        ));
    }

    Ok(ret)
}
