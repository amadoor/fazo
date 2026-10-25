/// The input must already satisfy `still_fails`; the result always does.
pub fn minimize(mut lines: Vec<String>, mut still_fails: impl FnMut(&[String]) -> bool) -> Vec<String> {
    let mut removed_any = true;
    while removed_any {
        removed_any = false;
        let mut index = 0;
        while index < lines.len() {
            let candidate: Vec<String> = lines[..index]
                .iter()
                .chain(&lines[index + 1..])
                .cloned()
                .collect();
            if still_fails(&candidate) {
                lines = candidate;
                removed_any = true;
            } else {
                index += 1;
            }
        }
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::minimize;

    #[test]
    fn shrinks_to_the_lines_the_mismatch_needs() {
        let lines: Vec<String> = (0..10)
            .map(|index| {
                if index == 6 {
                    "print(7)".to_string()
                } else {
                    format!("print({index})")
                }
            })
            .collect();
        let shrunk = minimize(lines, |candidate| {
            candidate.iter().any(|line| line == "print(7)")
        });
        assert_eq!(shrunk, vec!["print(7)".to_string()]);
    }

    #[test]
    fn keeps_lines_that_removal_would_break() {
        let lines: Vec<String> = ["local a = 1", "print(a)", "print(0)"]
            .into_iter()
            .map(String::from)
            .collect();
        let shrunk = minimize(lines, |candidate| {
            candidate.iter().any(|line| line == "local a = 1")
                && candidate.iter().any(|line| line == "print(a)")
        });
        assert_eq!(shrunk, vec!["local a = 1".to_string(), "print(a)".to_string()]);
    }
}
