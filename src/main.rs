use std::io::Write;
use std::path::PathBuf;

use umix_2024::*;

#[argopt::subcmd]
fn run(
    #[opt(long, default_value = "interpreter")] engine: String,
    bin: PathBuf,
) -> anyhow::Result<()> {
    let bin = std::fs::read(bin)?;

    match engine.as_ref() {
        "threaded" => {
            let mut um = threaded::Machine::from_bin(&bin)?;
            um.run();
        }
        "interpreter" => {
            let mut um = interpreter::Machine::from_bin(&bin)?;
            um.run();
        }
        "jit" => {
            let mut um = jit::Machine::from_bin(&bin)?;
            um.run();
        }
        _ => anyhow::bail!("unknown engine: {}", engine),
    }

    Ok(())
}

#[argopt::subcmd]
fn disasm(bin: PathBuf) -> anyhow::Result<()> {
    let bin = std::fs::read(bin)?;
    let asm = disasm::disasm(0, &disasm::decode(&bin)?)?;

    for instr in asm {
        println!("{}", instr);
    }

    Ok(())
}

#[argopt::cmd_group(commands = [run, disasm])]
fn main() -> anyhow::Result<()> {
    env_logger::builder()
        .format(|buf, record| writeln!(buf, "{}", record.args()))
        .init();
}
