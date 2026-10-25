#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum Listing {
    #[default]
    None,
    Code,
    Full,
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Flags {
    pub interactive: bool,
    pub version: bool,
    pub ignore_env: bool,
    pub execute: bool,
    pub warnings: bool,
    pub listing: Listing,
    pub opcode_counts: bool,
    pub extensions: Vec<Vec<u8>>,
}

/// `script` is 0 when argv names no script.
#[derive(Debug, PartialEq, Eq)]
pub enum Command {
    Run { flags: Flags, script: usize },
    BadOption(usize),
}

enum Step {
    Next,
    Argument,
    Stop(usize),
    Bad,
}

pub fn parse(argv: &[Vec<u8>]) -> Command {
    let mut flags = Flags::default();
    let mut index = 1;
    while let Some(argument) = argv.get(index) {
        if argument.first() != Some(&b'-') {
            return Command::Run { flags, script: index };
        }
        match option(&mut flags, argv, index) {
            Step::Next => index += 1,
            Step::Argument => index += 2,
            Step::Stop(script) => return Command::Run { flags, script },
            Step::Bad => return Command::BadOption(index),
        }
    }
    Command::Run { flags, script: 0 }
}

fn option(flags: &mut Flags, argv: &[Vec<u8>], index: usize) -> Step {
    let argument = argv[index].as_slice();
    let Some(&letter) = argument.get(1) else {
        return Step::Stop(index);
    };
    match letter {
        b'-' => double_dash(flags, argv, index),
        b'L' => listing(argument, &mut flags.listing),
        b'E' => set_alone(argument, &mut flags.ignore_env),
        b'W' => set_alone(argument, &mut flags.warnings),
        b'i' => {
            flags.interactive = true;
            set_alone(argument, &mut flags.version)
        }
        b'v' => set_alone(argument, &mut flags.version),
        b'e' => {
            flags.execute = true;
            with_argument(argv, index)
        }
        b'l' => with_argument(argv, index),
        b'X' => extension(argv, index, &mut flags.extensions),
        _ => Step::Bad,
    }
}

fn set_alone(argument: &[u8], flag: &mut bool) -> Step {
    if argument.len() != 2 {
        return Step::Bad;
    }
    *flag = true;
    Step::Next
}

fn with_argument(argv: &[Vec<u8>], index: usize) -> Step {
    if argv[index].len() > 2 {
        return Step::Next;
    }
    match argv.get(index + 1) {
        Some(next) if next.first() != Some(&b'-') => Step::Argument,
        _ => Step::Bad,
    }
}

fn extension(argv: &[Vec<u8>], index: usize, extensions: &mut Vec<Vec<u8>>) -> Step {
    if argv[index].len() != 2 {
        return Step::Bad;
    }
    let Some(option) = argv.get(index + 1) else {
        return Step::Bad;
    };
    if !fazo::Options::default().set_extension(option) {
        return Step::Bad;
    }
    extensions.push(option.clone());
    Step::Argument
}

fn listing(argument: &[u8], listing: &mut Listing) -> Step {
    *listing = match argument {
        b"-L" => Listing::Code,
        b"-LL" => Listing::Full,
        _ => return Step::Bad,
    };
    Step::Next
}

fn double_dash(flags: &mut Flags, argv: &[Vec<u8>], index: usize) -> Step {
    if cfg!(feature = "opcode-counts") && argv[index] == b"--opcode-counts" {
        flags.opcode_counts = true;
        return Step::Next;
    }
    stop_after_double_dash(argv, index)
}

fn stop_after_double_dash(argv: &[Vec<u8>], index: usize) -> Step {
    if argv[index].len() != 2 {
        return Step::Bad;
    }
    Step::Stop(if index + 1 < argv.len() { index + 1 } else { 0 })
}

#[cfg(test)]
mod tests {
    use super::{Command, Flags, Listing, parse};

    fn argv(words: &[&str]) -> Vec<Vec<u8>> {
        std::iter::once("fazo")
            .chain(words.iter().copied())
            .map(|word| word.as_bytes().to_vec())
            .collect()
    }

    #[test]
    fn the_first_word_that_is_not_an_option_is_the_script() {
        assert_eq!(
            parse(&argv(&["-e", "x", "t.lua", "-v"])),
            Command::Run {
                flags: Flags {
                    execute: true,
                    ..Flags::default()
                },
                script: 3,
            }
        );
    }

    #[test]
    fn attached_arguments_need_no_next_word() {
        assert_eq!(
            parse(&argv(&["-eprint(1)", "-lmod"])),
            Command::Run {
                flags: Flags {
                    execute: true,
                    ..Flags::default()
                },
                script: 0,
            }
        );
    }

    #[test]
    fn capital_l_lists_the_chunk_and_twice_lists_it_in_full() {
        assert_eq!(
            parse(&argv(&["-L", "t.lua"])),
            Command::Run {
                flags: Flags {
                    listing: Listing::Code,
                    ..Flags::default()
                },
                script: 2,
            }
        );
        assert_eq!(
            parse(&argv(&["-LL", "t.lua"])),
            Command::Run {
                flags: Flags {
                    listing: Listing::Full,
                    ..Flags::default()
                },
                script: 2,
            }
        );
        assert_eq!(parse(&argv(&["-LX", "t.lua"])), Command::BadOption(1));
    }

    #[test]
    fn an_option_without_its_argument_is_bad() {
        assert_eq!(parse(&argv(&["-e"])), Command::BadOption(1));
        assert_eq!(parse(&argv(&["-l", "-v"])), Command::BadOption(1));
    }

    #[test]
    fn a_double_dash_names_the_script_after_it() {
        assert_eq!(
            parse(&argv(&["--", "t.lua"])),
            Command::Run {
                flags: Flags::default(),
                script: 2,
            }
        );
        assert_eq!(
            parse(&argv(&["--"])),
            Command::Run {
                flags: Flags::default(),
                script: 0,
            }
        );
    }

    #[test]
    fn a_lone_dash_is_the_standard_input_script() {
        assert_eq!(
            parse(&argv(&["-"])),
            Command::Run {
                flags: Flags::default(),
                script: 1,
            }
        );
    }

    #[test]
    fn interactive_implies_version() {
        assert_eq!(
            parse(&argv(&["-i"])),
            Command::Run {
                flags: Flags {
                    interactive: true,
                    version: true,
                    ..Flags::default()
                },
                script: 0,
            }
        );
    }

    #[test]
    fn unknown_and_malformed_options_are_bad() {
        assert_eq!(parse(&argv(&["-x"])), Command::BadOption(1));
        assert_eq!(parse(&argv(&["--x"])), Command::BadOption(1));
        assert_eq!(parse(&argv(&["-Ex"])), Command::BadOption(1));
    }
}
