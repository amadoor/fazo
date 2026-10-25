use crate::vm::interrupt::Interrupts;

const ESCAPE: u8 = b'%';
const SPECIALS: &[u8] = b"^$*+?.([%-";
const MAX_CALL_DEPTH: usize = 200;
const MAX_CAPTURES: usize = 32;
const CHECK_STRIDE: usize = 1024;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum PatternError {
    Message(String),
    Interrupted,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Capture {
    Span { start: usize, end: usize },
    Position(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Match {
    pub start: usize,
    pub end: usize,
    pub captures: Vec<Capture>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotLength {
    Unfinished,
    Position,
    Span(usize),
}

#[derive(Clone, Copy, Debug)]
struct Slot {
    init: usize,
    length: SlotLength,
}

enum Step {
    Goto {
        subject_pos: usize,
        pattern_pos: usize,
    },
    Done(Option<usize>),
}

struct MatchState<'a> {
    subject: &'a [u8],
    pattern: &'a [u8],
    depth: usize,
    captures: Vec<Slot>,
    interrupts: &'a mut Interrupts,
}

impl<'a> MatchState<'a> {
    fn new(subject: &'a [u8], pattern: &'a [u8], interrupts: &'a mut Interrupts) -> Self {
        Self {
            subject,
            pattern,
            depth: MAX_CALL_DEPTH,
            captures: Vec::new(),
            interrupts,
        }
    }

    fn check_interrupt(&self) -> Result<(), PatternError> {
        if self.interrupts.raised() {
            return Err(PatternError::Interrupted);
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.depth = MAX_CALL_DEPTH;
        self.captures.clear();
    }

    fn subject_byte(&self, index: usize) -> u8 {
        self.subject.get(index).copied().unwrap_or(0)
    }

    fn pattern_byte(&self, index: usize) -> u8 {
        self.pattern.get(index).copied().unwrap_or(0)
    }

    fn match_at_position(&mut self, start: usize) -> Result<Option<Match>, PatternError> {
        self.reset();

        let Some(end) = self.match_here(start, 0)? else {
            return Ok(None);
        };
        let captures = self.collect_captures()?;
        Ok(Some(Match {
            start,
            end,
            captures,
        }))
    }

    fn collect_captures(&self) -> Result<Vec<Capture>, PatternError> {
        self.captures
            .iter()
            .map(|slot| match slot.length {
                SlotLength::Unfinished => Err(PatternError::Message("unfinished capture".to_string())),
                SlotLength::Position => Ok(Capture::Position(slot.init + 1)),
                SlotLength::Span(len) => Ok(Capture::Span {
                    start: slot.init,
                    end: slot.init + len,
                }),
            })
            .collect()
    }

    fn match_here(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Option<usize>, PatternError> {
        if self.interrupts.due() {
            return Err(PatternError::Interrupted);
        }
        if self.depth == 0 {
            return Err(PatternError::Message("pattern too complex".to_string()));
        }
        self.depth -= 1;

        let result = self.run(subject_pos, pattern_pos);
        self.depth += 1;
        result
    }

    fn run(&mut self, mut subject_pos: usize, mut pattern_pos: usize) -> Result<Option<usize>, PatternError> {
        loop {
            match self.step(subject_pos, pattern_pos)? {
                Step::Goto {
                    subject_pos: next_subject,
                    pattern_pos: next_pattern,
                } => {
                    subject_pos = next_subject;
                    pattern_pos = next_pattern;
                }
                Step::Done(result) => return Ok(result),
            }
        }
    }

    fn step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        if pattern_pos == self.pattern.len() {
            return Ok(Step::Done(Some(subject_pos)));
        }

        match self.pattern[pattern_pos] {
            b'(' => self.open_capture(subject_pos, pattern_pos),
            b')' => Ok(Step::Done(self.close_capture(subject_pos, pattern_pos + 1)?)),
            b'$' if pattern_pos + 1 == self.pattern.len() => {
                Ok(Step::Done(self.match_end(subject_pos)))
            }
            ESCAPE => self.escape_step(subject_pos, pattern_pos),
            _ => self.default_step(subject_pos, pattern_pos),
        }
    }

    fn match_end(&self, subject_pos: usize) -> Option<usize> {
        (subject_pos == self.subject.len()).then_some(subject_pos)
    }

    fn open_capture(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        let (body, length) = if self.pattern_byte(pattern_pos + 1) == b')' {
            (pattern_pos + 2, SlotLength::Position)
        } else {
            (pattern_pos + 1, SlotLength::Unfinished)
        };
        Ok(Step::Done(self.start_capture(subject_pos, body, length)?))
    }

    fn start_capture(
        &mut self,
        subject_pos: usize,
        pattern_pos: usize,
        length: SlotLength,
    ) -> Result<Option<usize>, PatternError> {
        if self.captures.len() >= MAX_CAPTURES {
            return Err(PatternError::Message("too many captures".to_string()));
        }
        self.captures.push(Slot {
            init: subject_pos,
            length,
        });

        let result = self.match_here(subject_pos, pattern_pos)?;
        if result.is_none() {
            self.captures.pop();
        }
        Ok(result)
    }

    fn close_capture(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Option<usize>, PatternError> {
        let index = self.capture_to_close()?;
        let init = self.captures[index].init;
        self.captures[index].length = SlotLength::Span(subject_pos - init);

        let result = self.match_here(subject_pos, pattern_pos)?;
        if result.is_none() {
            self.captures[index].length = SlotLength::Unfinished;
        }
        Ok(result)
    }

    fn capture_to_close(&self) -> Result<usize, PatternError> {
        self.captures
            .iter()
            .rposition(|slot| slot.length == SlotLength::Unfinished)
            .ok_or_else(|| PatternError::Message("invalid pattern capture".to_string()))
    }

    fn escape_step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        match self.pattern_byte(pattern_pos + 1) {
            b'b' => self.balance_step(subject_pos, pattern_pos),
            b'f' => self.frontier_step(subject_pos, pattern_pos),
            b'0'..=b'9' => self.back_reference_step(subject_pos, pattern_pos),
            _ => self.default_step(subject_pos, pattern_pos),
        }
    }

    fn balance_step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        match self.match_balance(subject_pos, pattern_pos + 2)? {
            Some(end) => Ok(Step::Goto {
                subject_pos: end,
                pattern_pos: pattern_pos + 4,
            }),
            None => Ok(Step::Done(None)),
        }
    }

    fn match_balance(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Option<usize>, PatternError> {
        if pattern_pos + 1 >= self.pattern.len() {
            return Err(PatternError::Message("malformed pattern (missing arguments to '%b')".to_string()));
        }

        let open = self.pattern[pattern_pos];
        let close = self.pattern[pattern_pos + 1];
        if self.subject.get(subject_pos) != Some(&open) {
            return Ok(None);
        }

        let mut depth = 1usize;
        let mut start = subject_pos + 1;
        while start < self.subject.len() {
            let end = (start + CHECK_STRIDE).min(self.subject.len());
            for index in start..end {
                let byte = self.subject[index];
                if byte == close {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(Some(index + 1));
                    }
                } else if byte == open {
                    depth += 1;
                }
            }
            start = end;
            self.check_interrupt()?;
        }
        Ok(None)
    }

    fn frontier_step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        let set = pattern_pos + 2;
        if self.pattern_byte(set) != b'[' {
            return Err(PatternError::Message("missing '[' after '%f' in pattern".to_string()));
        }

        let set_end = self.class_end(set)?;
        let close = set_end - 1;
        let previous = if subject_pos == 0 {
            0
        } else {
            self.subject[subject_pos - 1]
        };
        let current = self.subject_byte(subject_pos);
        if !bracket_class(self.pattern, previous, set, close) && bracket_class(self.pattern, current, set, close) {
            return Ok(Step::Goto {
                subject_pos,
                pattern_pos: set_end,
            });
        }
        Ok(Step::Done(None))
    }

    fn back_reference_step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        let digit = self.pattern_byte(pattern_pos + 1);
        match self.match_capture(subject_pos, digit)? {
            Some(end) => Ok(Step::Goto {
                subject_pos: end,
                pattern_pos: pattern_pos + 2,
            }),
            None => Ok(Step::Done(None)),
        }
    }

    fn match_capture(&self, subject_pos: usize, digit: u8) -> Result<Option<usize>, PatternError> {
        let index = self.check_capture(digit)?;
        let slot = self.captures[index];
        let SlotLength::Span(len) = slot.length else {
            return Ok(None);
        };

        if self.subject.len() - subject_pos < len {
            return Ok(None);
        }
        let captured = &self.subject[slot.init..slot.init + len];
        if captured == &self.subject[subject_pos..subject_pos + len] {
            Ok(Some(subject_pos + len))
        } else {
            Ok(None)
        }
    }

    fn check_capture(&self, digit: u8) -> Result<usize, PatternError> {
        let number = usize::from(digit - b'0');
        match number.checked_sub(1) {
            Some(index)
                if index < self.captures.len()
                    && self.captures[index].length != SlotLength::Unfinished =>
            {
                Ok(index)
            }
            _ => Err(PatternError::Message(format!("invalid capture index %{number}"))),
        }
    }

    fn default_step(&mut self, subject_pos: usize, pattern_pos: usize) -> Result<Step, PatternError> {
        let class_end = self.class_end(pattern_pos)?;
        let suffix = self.pattern_byte(class_end);

        if self.single_match(subject_pos, pattern_pos, class_end) {
            return self.repetition_step(subject_pos, pattern_pos, class_end, suffix);
        }
        if matches!(suffix, b'*' | b'?' | b'-') {
            return Ok(Step::Goto {
                subject_pos,
                pattern_pos: class_end + 1,
            });
        }
        Ok(Step::Done(None))
    }

    fn repetition_step(
        &mut self,
        subject_pos: usize,
        class_start: usize,
        class_end: usize,
        suffix: u8,
    ) -> Result<Step, PatternError> {
        match suffix {
            b'?' => match self.match_here(subject_pos + 1, class_end + 1)? {
                Some(matched) => Ok(Step::Done(Some(matched))),
                None => Ok(Step::Goto {
                    subject_pos,
                    pattern_pos: class_end + 1,
                }),
            },
            b'+' => Ok(Step::Done(self.max_expand(
                subject_pos + 1,
                class_start,
                class_end,
            )?)),
            b'*' => Ok(Step::Done(self.max_expand(
                subject_pos,
                class_start,
                class_end,
            )?)),
            b'-' => Ok(Step::Done(self.min_expand(
                subject_pos,
                class_start,
                class_end,
            )?)),
            _ => Ok(Step::Goto {
                subject_pos: subject_pos + 1,
                pattern_pos: class_end,
            }),
        }
    }

    fn max_expand(
        &mut self,
        subject_pos: usize,
        class_start: usize,
        class_end: usize,
    ) -> Result<Option<usize>, PatternError> {
        let mut count = self.count_matches(subject_pos, class_start, class_end)?;
        loop {
            if let Some(matched) = self.match_here(subject_pos + count, class_end + 1)? {
                return Ok(Some(matched));
            }
            if count == 0 {
                return Ok(None);
            }
            count -= 1;
        }
    }

    fn count_matches(&self, subject_pos: usize, class_start: usize, class_end: usize) -> Result<usize, PatternError> {
        let mut count = 0;
        loop {
            let stride_end = count + CHECK_STRIDE;
            while count < stride_end && self.single_match(subject_pos + count, class_start, class_end) {
                count += 1;
            }
            if count < stride_end {
                return Ok(count);
            }
            self.check_interrupt()?;
        }
    }

    fn min_expand(
        &mut self,
        mut subject_pos: usize,
        class_start: usize,
        class_end: usize,
    ) -> Result<Option<usize>, PatternError> {
        loop {
            if let Some(matched) = self.match_here(subject_pos, class_end + 1)? {
                return Ok(Some(matched));
            }
            if !self.single_match(subject_pos, class_start, class_end) {
                return Ok(None);
            }
            subject_pos += 1;
        }
    }

    fn single_match(&self, subject_pos: usize, pattern_pos: usize, class_end: usize) -> bool {
        let Some(&byte) = self.subject.get(subject_pos) else {
            return false;
        };
        item_matches(self.pattern, byte, pattern_pos, class_end)
    }

    fn class_end(&self, pattern_pos: usize) -> Result<usize, PatternError> {
        let next = pattern_pos + 1;
        match self.pattern[pattern_pos] {
            ESCAPE if next == self.pattern.len() => {
                Err(PatternError::Message("malformed pattern (ends with '%')".to_string()))
            }
            ESCAPE => Ok(next + 1),
            b'[' => self.set_end(next),
            _ => Ok(next),
        }
    }

    fn set_end(&self, mut index: usize) -> Result<usize, PatternError> {
        if self.pattern_byte(index) == b'^' {
            index += 1;
        }

        loop {
            if index == self.pattern.len() {
                return Err(PatternError::Message("malformed pattern (missing ']')".to_string()));
            }
            let byte = self.pattern[index];
            index += 1;
            if byte == ESCAPE && index < self.pattern.len() {
                index += 1;
            }
            if self.pattern_byte(index) == b']' {
                return Ok(index + 1);
            }
        }
    }

}

fn byte_at(bytes: &[u8], index: usize) -> u8 {
    bytes.get(index).copied().unwrap_or(0)
}

/// Follows `singlematch` in lstrlib.c.
fn item_matches(pattern: &[u8], byte: u8, pattern_pos: usize, class_end: usize) -> bool {
    match pattern[pattern_pos] {
        b'.' => true,
        ESCAPE => class_matches(byte, byte_at(pattern, pattern_pos + 1)),
        b'[' => bracket_class(pattern, byte, pattern_pos, class_end - 1),
        literal => literal == byte,
    }
}

fn bracket_class(pattern: &[u8], byte: u8, open: usize, close: usize) -> bool {
    let negated = byte_at(pattern, open + 1) == b'^';
    let mut index = if negated { open + 1 } else { open };

    loop {
        index += 1;
        if index >= close {
            return negated;
        }

        if pattern[index] == ESCAPE {
            index += 1;
            if class_matches(byte, byte_at(pattern, index)) {
                return !negated;
            }
        } else if byte_at(pattern, index + 1) == b'-' && index + 2 < close {
            index += 2;
            if pattern[index - 2] <= byte && byte <= pattern[index] {
                return !negated;
            }
        } else if pattern[index] == byte {
            return !negated;
        }
    }
}

/// A malformed item is left to the matcher, which reports it as `do_match` does.
pub(crate) fn required_lead(pattern: &[u8]) -> Option<(usize, usize)> {
    let class_end = match *pattern.first()? {
        b'(' | b')' => return None,
        b'$' if pattern.len() == 1 => return None,
        ESCAPE => match *pattern.get(1)? {
            b'b' | b'f' | b'0'..=b'9' => return None,
            _ => 2,
        },
        b'[' => lead_set_end(pattern)?,
        _ => 1,
    };
    if matches!(byte_at(pattern, class_end), b'*' | b'?' | b'-') {
        return None;
    }
    Some((0, class_end))
}

fn lead_set_end(pattern: &[u8]) -> Option<usize> {
    let mut index = 1;
    if byte_at(pattern, index) == b'^' {
        index += 1;
    }
    loop {
        if index == pattern.len() {
            return None;
        }
        let byte = pattern[index];
        index += 1;
        if byte == ESCAPE && index < pattern.len() {
            index += 1;
        }
        if byte_at(pattern, index) == b']' {
            return Some(index + 1);
        }
    }
}

pub(crate) fn next_start(subject: &[u8], pattern: &[u8], lead: Option<(usize, usize)>, from: usize) -> usize {
    let mut position = from;
    while position <= subject.len() && lead_rejects(subject, pattern, lead, position) {
        position += 1;
    }
    position
}

/// Skipping such a position changes no result or error, since the match fails at that item.
fn lead_rejects(subject: &[u8], pattern: &[u8], lead: Option<(usize, usize)>, position: usize) -> bool {
    let Some((start, end)) = lead else {
        return false;
    };
    match subject.get(position) {
        Some(&byte) => !item_matches(pattern, byte, start, end),
        None => true,
    }
}

fn class_matches(byte: u8, class: u8) -> bool {
    let matched = match class.to_ascii_lowercase() {
        b'a' => byte.is_ascii_alphabetic(),
        b'c' => byte.is_ascii_control(),
        b'd' => byte.is_ascii_digit(),
        b'g' => byte.is_ascii_graphic(),
        b'l' => byte.is_ascii_lowercase(),
        b'p' => byte.is_ascii_punctuation(),
        b's' => matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r'),
        b'u' => byte.is_ascii_uppercase(),
        b'w' => byte.is_ascii_alphanumeric(),
        b'x' => byte.is_ascii_hexdigit(),
        b'z' => byte == 0,
        _ => return class == byte,
    };
    if class.is_ascii_lowercase() {
        matched
    } else {
        !matched
    }
}

pub(crate) fn search(subject: &[u8], pattern: &[u8], init: usize, interrupts: &mut Interrupts) -> Result<Option<Match>, PatternError> {
    let anchored = pattern.first() == Some(&b'^');
    let body = if anchored { &pattern[1..] } else { pattern };
    let last = if anchored { init } else { subject.len() };

    if init > subject.len() {
        return Ok(None);
    }

    let lead = required_lead(body);
    let mut state = MatchState::new(subject, body, interrupts);
    for start in init..=last {
        if lead_rejects(subject, body, lead, start) {
            continue;
        }
        if let Some(found) = state.match_at_position(start)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}

pub(crate) fn match_at(subject: &[u8], pattern: &[u8], position: usize, interrupts: &mut Interrupts) -> Result<Option<Match>, PatternError> {
    if position > subject.len() {
        return Ok(None);
    }

    let mut state = MatchState::new(subject, pattern, interrupts);
    state.match_at_position(position)
}

pub(crate) fn is_plain(pattern: &[u8]) -> bool {
    !pattern.iter().any(|byte| SPECIALS.contains(byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole(subject: &[u8], pattern: &[u8]) -> Result<Option<(usize, usize)>, PatternError> {
        Ok(search(subject, pattern, 0, &mut Interrupts::never())?.map(|found| (found.start, found.end)))
    }

    #[test]
    fn literal_match_reports_whole_range() {
        assert_eq!(whole(b"hello world", b"lo w"), Ok(Some((3, 7))));
    }

    #[test]
    fn literal_without_occurrence_is_none() {
        assert_eq!(whole(b"abc", b"d"), Ok(None));
    }

    #[test]
    fn empty_pattern_matches_empty_at_start() {
        assert_eq!(whole(b"abc", b""), Ok(Some((0, 0))));
        assert_eq!(whole(b"", b""), Ok(Some((0, 0))));
    }

    #[test]
    fn dollar_is_literal_unless_last() {
        assert_eq!(whole(b"a$b", b"$b"), Ok(Some((1, 3))));
    }

    #[test]
    fn dollar_anchors_end_of_subject() {
        assert_eq!(whole(b"ab", b"b$"), Ok(Some((1, 2))));
        assert_eq!(whole(b"ba", b"b$"), Ok(None));
    }

    #[test]
    fn quantifiers_expand_greedy_optional_and_lazy() {
        assert_eq!(whole(b"aaab", b"a*b"), Ok(Some((0, 4))));
        assert_eq!(whole(b"aaab", b"a+"), Ok(Some((0, 3))));
        assert_eq!(whole(b"b", b"a?b"), Ok(Some((0, 1))));
        assert_eq!(whole(b"ab", b"a-b"), Ok(Some((0, 2))));
    }

    #[test]
    fn letter_and_digit_classes() {
        assert_eq!(whole(b"12ab", b"%a+"), Ok(Some((2, 4))));
        assert_eq!(whole(b"ab12", b"%d"), Ok(Some((2, 3))));
    }

    #[test]
    fn upper_case_classes_are_complements() {
        assert_eq!(whole(b"ab12", b"%D+"), Ok(Some((0, 2))));
        assert_eq!(whole(b"ab1", b"%A"), Ok(Some((2, 3))));
    }

    #[test]
    fn remaining_classes_match_ascii_ctype() {
        assert_eq!(whole(b"a\tb", b"%s"), Ok(Some((1, 2))));
        assert_eq!(whole(b"a\x01b", b"%c"), Ok(Some((1, 2))));
        assert_eq!(whole(b" x", b"%g"), Ok(Some((1, 2))));
        assert_eq!(whole(b"A b", b"%l"), Ok(Some((2, 3))));
        assert_eq!(whole(b"aB", b"%u"), Ok(Some((1, 2))));
        assert_eq!(whole(b"a,b", b"%p"), Ok(Some((1, 2))));
        assert_eq!(whole(b"-a1", b"%w+"), Ok(Some((1, 3))));
        assert_eq!(whole(b"zfF0g", b"%x+"), Ok(Some((1, 4))));
        assert_eq!(whole(b"a\0b", b"%z"), Ok(Some((1, 2))));
    }

    #[test]
    fn bytes_above_127_belong_to_no_class() {
        assert_eq!(whole(&[0xC3, 0xA9], b"%w"), Ok(None));
        assert_eq!(whole(&[0xC3, 0xA9], b"%a"), Ok(None));
        assert_eq!(whole(&[0xC3, 0xA9], b"."), Ok(Some((0, 1))));
    }

    #[test]
    fn sets_with_ranges() {
        assert_eq!(whole(b"xbz", b"[a-c]+"), Ok(Some((1, 2))));
    }

    #[test]
    fn sets_with_leading_bracket_and_escapes() {
        assert_eq!(whole(b"x]y", b"[]]"), Ok(Some((1, 2))));
        assert_eq!(whole(b"a-b", b"[%-]"), Ok(Some((1, 2))));
        assert_eq!(whole(b"a1 ", b"[%d ]+"), Ok(Some((1, 3))));
    }

    #[test]
    fn complemented_set() {
        assert_eq!(whole(b"abc1", b"[^a-z]"), Ok(Some((3, 4))));
    }

    #[test]
    fn balanced_match() {
        assert_eq!(whole(b"x(a(b)c)y", b"%b()"), Ok(Some((1, 8))));
        assert_eq!(whole(b"(a", b"%b()"), Ok(None));
    }

    #[test]
    fn frontier_matches_class_transitions() {
        assert_eq!(whole(b"THE (quick) fox", b"%f[%a]%a+"), Ok(Some((0, 3))));
        assert_eq!(whole(b"ab cd", b"%f[%s]"), Ok(Some((2, 2))));
    }

    #[test]
    fn anchored_search_only_tries_init() {
        assert_eq!(search(b"hello", b"^ell", 0, &mut Interrupts::never()), Ok(None));
        assert_eq!(
            search(b"hello", b"^he", 0, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 0,
                end: 2,
                captures: vec![],
            }))
        );
        assert_eq!(
            search(b"hello", b"^l", 2, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 2,
                end: 3,
                captures: vec![],
            }))
        );
    }

    #[test]
    fn search_starts_at_init() {
        assert_eq!(
            search(b"abcabc", b"b", 2, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 4,
                end: 5,
                captures: vec![],
            }))
        );
        assert_eq!(search(b"ab", b"", 3, &mut Interrupts::never()), Ok(None));
    }

    #[test]
    fn match_at_treats_caret_as_literal() {
        assert_eq!(match_at(b"abc", b"^a", 0, &mut Interrupts::never()), Ok(None));
        assert_eq!(
            match_at(b"abc", b"b", 1, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 1,
                end: 2,
                captures: vec![],
            }))
        );
    }

    #[test]
    fn captures_are_reported_in_order() {
        assert_eq!(
            search(b"key=val", b"(%w+)=(%w+)", 0, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 0,
                end: 7,
                captures: vec![
                    Capture::Span { start: 0, end: 3 },
                    Capture::Span { start: 4, end: 7 },
                ],
            }))
        );
    }

    #[test]
    fn position_captures_are_one_based() {
        assert_eq!(
            search(b"abc", b"()b()", 0, &mut Interrupts::never()),
            Ok(Some(Match {
                start: 1,
                end: 2,
                captures: vec![Capture::Position(2), Capture::Position(3)],
            }))
        );
    }

    #[test]
    fn back_reference_repeats_capture() {
        assert_eq!(whole(b"abcabc", b"(abc)%1"), Ok(Some((0, 6))));
        assert_eq!(whole(b"abcabd", b"(abc)%1"), Ok(None));
    }

    #[test]
    fn back_reference_to_position_capture_never_matches() {
        assert_eq!(whole(b"aa", b"()%1"), Ok(None));
    }

    #[test]
    fn is_plain_detects_specials() {
        assert!(is_plain(b"hello"));
        assert!(is_plain(b"a]b"));
        assert!(!is_plain(b"a.b"));
        assert!(!is_plain(b"%"));
    }

    #[test]
    fn malformed_escape_at_end() {
        assert_eq!(
            whole(b"a", b"a%"),
            Err(PatternError::Message("malformed pattern (ends with '%')".to_string()))
        );
    }

    #[test]
    fn unclosed_set() {
        assert_eq!(
            whole(b"a", b"[a"),
            Err(PatternError::Message("malformed pattern (missing ']')".to_string()))
        );
    }

    #[test]
    fn balance_needs_two_arguments() {
        assert_eq!(
            whole(b"a", b"a%b"),
            Err(PatternError::Message("malformed pattern (missing arguments to '%b')".to_string()))
        );
    }

    #[test]
    fn frontier_needs_set() {
        assert_eq!(
            whole(b"a", b"%fa"),
            Err(PatternError::Message("missing '[' after '%f' in pattern".to_string()))
        );
    }

    #[test]
    fn back_reference_needs_finished_capture() {
        assert_eq!(
            whole(b"a", b"%1"),
            Err(PatternError::Message("invalid capture index %1".to_string()))
        );
        assert_eq!(
            whole(b"a", b"%0"),
            Err(PatternError::Message("invalid capture index %0".to_string()))
        );
        assert_eq!(
            whole(b"aa", b"(a%1)"),
            Err(PatternError::Message("invalid capture index %1".to_string()))
        );
    }

    #[test]
    fn close_without_open() {
        assert_eq!(
            whole(b"a", b"a)"),
            Err(PatternError::Message("invalid pattern capture".to_string()))
        );
    }

    #[test]
    fn unfinished_capture_is_an_error_on_result() {
        assert_eq!(
            search(b"a", b"(a", 0, &mut Interrupts::never()),
            Err(PatternError::Message("unfinished capture".to_string()))
        );
    }

    #[test]
    fn too_many_captures() {
        let pattern = "(".repeat(33);
        assert_eq!(
            search(b"", pattern.as_bytes(), 0, &mut Interrupts::never()),
            Err(PatternError::Message("too many captures".to_string()))
        );
    }

    #[test]
    fn pattern_nesting_limit() {
        let pattern = "a?".repeat(250);
        let subject = "a".repeat(250);
        assert_eq!(
            search(subject.as_bytes(), pattern.as_bytes(), 0, &mut Interrupts::never()),
            Err(PatternError::Message("pattern too complex".to_string()))
        );
    }
}
