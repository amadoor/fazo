//! Follows PUC Lua 5.5.1's llex.c (MIT, Copyright (C) 1994-2025 Lua.org, PUC-Rio).
//! Names and strings are interned as they are read, so the parser compares them by reference.

use hashbrown::HashMap;

use super::Result;
use super::numeral::{self, Numeral};
use super::token::{self, Kind, Token, Value, hex_value};
use crate::{arena::Ref, heap::Heap, heap::SHORT_STRING_LIMIT, object::LuaString};

const MAX_LINE: u32 = i32::MAX as u32;
const MAX_CODE_POINT: u64 = 0x7FFF_FFFF;

/// Follows PUC's scanner anchor table: long strings are unique per chunk.
#[derive(Clone, Default)]
pub(crate) struct Strings {
    long: HashMap<Box<[u8]>, Ref<LuaString>>,
}

impl Strings {
    pub(crate) fn intern(&mut self, heap: &mut Heap, bytes: &[u8]) -> Ref<LuaString> {
        if bytes.len() <= SHORT_STRING_LIMIT {
            return heap.string(bytes);
        }
        if let Some(string) = self.long.get(bytes) {
            return *string;
        }
        let string = heap.string(bytes);
        self.long.insert(bytes.into(), string);
        string
    }

    pub(crate) fn clear(&mut self) {
        self.long.clear();
    }
}

pub(crate) struct Sink<'s> {
    pub heap: &'s mut Heap,
    pub strings: &'s mut Strings,
}

impl Sink<'_> {
    fn intern(&mut self, bytes: &[u8]) -> Ref<LuaString> {
        self.strings.intern(self.heap, bytes)
    }
}

/// A name's text is the range `name` of the source, not `buffer`.
pub(crate) struct Lexer<'a> {
    chunk: &'a [u8],
    source: &'a [u8],
    index: usize,
    current: Option<u8>,
    pub(crate) line: u32,
    buffer: Vec<u8>,
    name: Option<(usize, usize)>,
}

impl<'a> Lexer<'a> {
    pub(crate) fn new(chunk: &'a [u8], source: &'a [u8], mut buffer: Vec<u8>) -> Self {
        buffer.clear();
        let mut lexer = Self {
            chunk,
            source,
            index: 0,
            current: None,
            line: 1,
            buffer,
            name: None,
        };
        lexer.advance();
        lexer
    }

    pub(crate) fn into_buffer(self) -> Vec<u8> {
        self.buffer
    }

    pub(crate) fn message(&self, text: &str, near: Option<Vec<u8>>) -> Vec<u8> {
        self.message_at(self.line, text, near)
    }

    pub(crate) fn message_at(&self, line: u32, text: &str, near: Option<Vec<u8>>) -> Vec<u8> {
        let mut message = self.chunk.to_vec();
        message.extend_from_slice(format!(":{line}: {text}").as_bytes());
        if let Some(near) = near {
            message.extend_from_slice(b" near ");
            message.extend_from_slice(&near);
        }
        message
    }

    /// Follows `txtToken` in llex.c; the lookahead token's text is shown when one was read.
    pub(crate) fn near(&self, kind: Kind) -> Vec<u8> {
        match (kind, self.name) {
            (Kind::Name | Kind::Str | Kind::Int | Kind::Float, Some((start, end))) => token::quoted(&self.source[start..end]),
            (Kind::Name | Kind::Str | Kind::Int | Kind::Float, None) => token::quoted(&self.buffer),
            (kind, _) => token::describe(kind).into_bytes(),
        }
    }

    fn fail<T>(&self, text: &str, near: Option<Vec<u8>>) -> Result<T> {
        Err(self.message(text, near))
    }

    fn buffer_near(&self) -> Option<Vec<u8>> {
        Some(token::quoted(&self.buffer))
    }

    fn eof_near() -> Option<Vec<u8>> {
        Some(b"<eof>".to_vec())
    }

    #[inline]
    fn advance(&mut self) {
        self.current = self.source.get(self.index).copied();
        self.index += 1;
    }

    fn is_current(&self, byte: u8) -> bool {
        self.current == Some(byte)
    }

    #[inline]
    fn save_and_next(&mut self) {
        if let Some(byte) = self.current {
            self.buffer.push(byte);
        }
        self.advance();
    }

    fn check_next1(&mut self, byte: u8) -> bool {
        let matched = self.is_current(byte);
        if matched {
            self.advance();
        }
        matched
    }

    fn check_next2(&mut self, options: [u8; 2]) -> bool {
        let matched = self.current.is_some_and(|byte| options.contains(&byte));
        if matched {
            self.save_and_next();
        }
        matched
    }

    fn remove_saved(&mut self, count: usize) {
        let length = self.buffer.len();
        self.buffer.truncate(length.saturating_sub(count));
    }

    fn replace_escape(&mut self, byte: u8) {
        self.buffer.pop();
        self.buffer.push(byte);
    }

    fn inc_line(&mut self) -> Result<()> {
        let old = self.current;
        self.advance();
        if is_newline(self.current) && self.current != old {
            self.advance();
        }
        self.line += 1;
        if self.line >= MAX_LINE {
            return self.fail("chunk has too many lines", None);
        }
        Ok(())
    }

    pub(crate) fn lex(&mut self, sink: &mut Sink<'_>) -> Result<Token> {
        self.buffer.clear();
        self.name = None;
        loop {
            match self.current {
                None => return Ok(Token::new(Kind::Eos)),
                Some(b'\n' | b'\r') => self.inc_line()?,
                Some(b' ' | 0x0c | b'\t' | 0x0b) => self.advance(),
                Some(b'-') => {
                    self.advance();
                    if !self.is_current(b'-') {
                        return Ok(Token::new(Kind::Char(b'-')));
                    }
                    self.advance();
                    self.skip_comment()?;
                }
                Some(b'[') => return self.open_bracket(sink),
                Some(b'=') => return Ok(self.symbol(b'=', Kind::EqEq, b'=')),
                Some(b'<') => return Ok(self.less_or_greater(b'<', Kind::LessEq, Kind::Shl)),
                Some(b'>') => return Ok(self.less_or_greater(b'>', Kind::GreaterEq, Kind::Shr)),
                Some(b'/') => return Ok(self.symbol(b'/', Kind::IntDiv, b'/')),
                Some(b'~') => return Ok(self.symbol(b'=', Kind::NotEq, b'~')),
                Some(b':') => return Ok(self.symbol(b':', Kind::DbColon, b':')),
                Some(quote @ (b'"' | b'\'')) => return self.read_string(quote, sink),
                Some(b'.') => return self.dot(),
                Some(b'0'..=b'9') => return self.read_numeral(),
                Some(byte) if is_alpha(byte) => return Ok(self.name(sink)),
                Some(byte) => {
                    self.advance();
                    return Ok(Token::new(Kind::Char(byte)));
                }
            }
        }
    }

    fn symbol(&mut self, second: u8, double: Kind, single: u8) -> Token {
        self.advance();
        let kind = if self.check_next1(second) {
            double
        } else {
            Kind::Char(single)
        };
        Token::new(kind)
    }

    fn less_or_greater(&mut self, operator: u8, orequal: Kind, shift: Kind) -> Token {
        self.advance();
        let kind = if self.check_next1(b'=') {
            orequal
        } else if self.check_next1(operator) {
            shift
        } else {
            Kind::Char(operator)
        };
        Token::new(kind)
    }

    fn name(&mut self, sink: &mut Sink<'_>) -> Token {
        let start = self.index - 1;
        let mut end = self.index;
        while self.source.get(end).copied().is_some_and(is_alnum) {
            end += 1;
        }
        self.name = Some((start, end));
        self.index = end;
        self.advance();
        let text = &self.source[start..end];
        if let Some(kind) = token::keyword(text) {
            return Token::new(kind);
        }
        Token {
            kind: Kind::Name,
            value: Value::Str(sink.intern(text)),
        }
    }

    fn dot(&mut self) -> Result<Token> {
        self.save_and_next();
        if self.check_next1(b'.') {
            let kind = if self.check_next1(b'.') {
                Kind::Dots
            } else {
                Kind::Concat
            };
            return Ok(Token::new(kind));
        }
        if !self.current.is_some_and(|byte| byte.is_ascii_digit()) {
            return Ok(Token::new(Kind::Char(b'.')));
        }
        self.read_numeral()
    }

    fn read_numeral(&mut self) -> Result<Token> {
        let first = self.current;
        let mut exponent = *b"Ee";
        self.save_and_next();
        if first == Some(b'0') && self.check_next2(*b"xX") {
            exponent = *b"Pp";
        }
        loop {
            if self.check_next2(exponent) {
                self.check_next2(*b"-+");
            } else if self.current.is_some_and(|byte| byte.is_ascii_hexdigit() || byte == b'.') {
                self.save_and_next();
            } else {
                break;
            }
        }
        if self.current.is_some_and(is_alpha) {
            self.save_and_next();
        }
        let (kind, value) = match numeral::str2number(&self.buffer) {
            Some(Numeral::Int(value)) => (Kind::Int, Value::Int(value)),
            Some(Numeral::Float(value)) => (Kind::Float, Value::Float(value)),
            None => return self.fail("malformed number", self.buffer_near()),
        };
        Ok(Token { kind, value })
    }

    fn skip_separator(&mut self) -> usize {
        let delimiter = self.current;
        let mut count = 0;
        self.save_and_next();
        while self.is_current(b'=') {
            self.save_and_next();
            count += 1;
        }
        if self.current == delimiter {
            count + 2
        } else if count == 0 {
            1
        } else {
            0
        }
    }

    fn open_bracket(&mut self, sink: &mut Sink<'_>) -> Result<Token> {
        let separator = self.skip_separator();
        if separator >= 2 {
            self.read_long_string(false, separator)?;
            let value = sink.intern(&self.buffer[separator..self.buffer.len() - separator]);
            return Ok(Token {
                kind: Kind::Str,
                value: Value::Str(value),
            });
        }
        if separator == 0 {
            return self.fail("invalid long string delimiter", self.buffer_near());
        }
        Ok(Token::new(Kind::Char(b'[')))
    }

    fn skip_comment(&mut self) -> Result<()> {
        if self.is_current(b'[') {
            let separator = self.skip_separator();
            self.buffer.clear();
            if separator >= 2 {
                self.read_long_string(true, separator)?;
                self.buffer.clear();
                return Ok(());
            }
        }
        while self.current.is_some_and(|byte| !is_newline_byte(byte)) {
            self.advance();
        }
        Ok(())
    }

    fn read_long_string(&mut self, comment: bool, separator: usize) -> Result<()> {
        let start = self.line;
        self.save_and_next();
        if is_newline(self.current) {
            self.inc_line()?;
        }
        loop {
            match self.current {
                None => {
                    let what = if comment { "comment" } else { "string" };
                    let text = format!("unfinished long {what} (starting at line {start})");
                    return self.fail(&text, Self::eof_near());
                }
                Some(b']') => {
                    if self.skip_separator() == separator {
                        self.save_and_next();
                        return Ok(());
                    }
                }
                Some(b'\n' | b'\r') => {
                    self.buffer.push(b'\n');
                    self.inc_line()?;
                    if comment {
                        self.buffer.clear();
                    }
                }
                Some(_) => {
                    if comment {
                        self.advance();
                    } else {
                        self.save_and_next();
                    }
                }
            }
        }
    }

    fn read_string(&mut self, delimiter: u8, sink: &mut Sink<'_>) -> Result<Token> {
        self.save_and_next();
        while self.current != Some(delimiter) {
            match self.current {
                None => return self.fail("unfinished string", Self::eof_near()),
                Some(b'\n' | b'\r') => return self.fail("unfinished string", self.buffer_near()),
                Some(b'\\') => self.read_escape()?,
                Some(_) => self.save_and_next(),
            }
        }
        self.save_and_next();
        let value = sink.intern(&self.buffer[1..self.buffer.len() - 1]);
        Ok(Token {
            kind: Kind::Str,
            value: Value::Str(value),
        })
    }

    fn read_escape(&mut self) -> Result<()> {
        self.save_and_next();
        let Some(current) = self.current else {
            return Ok(());
        };
        if let Some(byte) = simple_escape(current) {
            self.advance();
            self.replace_escape(byte);
            return Ok(());
        }
        match current {
            b'x' => {
                let byte = self.read_hex_escape()?;
                self.advance();
                self.replace_escape(byte);
            }
            b'u' => self.read_utf8_escape()?,
            b'\n' | b'\r' => {
                self.inc_line()?;
                self.replace_escape(b'\n');
            }
            b'z' => self.skip_space_escape()?,
            _ => {
                self.escape_check(current.is_ascii_digit(), "invalid escape sequence")?;
                let byte = self.read_decimal_escape()?;
                self.replace_escape(byte);
            }
        }
        Ok(())
    }

    fn escape_check(&mut self, valid: bool, text: &str) -> Result<()> {
        if valid {
            return Ok(());
        }
        if self.current.is_some() {
            self.save_and_next();
        }
        self.fail(text, self.buffer_near())
    }

    fn hex_digit(&mut self) -> Result<u8> {
        self.save_and_next();
        let valid = self.current.is_some_and(|byte| byte.is_ascii_hexdigit());
        self.escape_check(valid, "hexadecimal digit expected")?;
        Ok(self.current.map_or(0, hex_value))
    }

    fn read_hex_escape(&mut self) -> Result<u8> {
        let high = self.hex_digit()?;
        let low = self.hex_digit()?;
        self.remove_saved(2);
        Ok((high << 4) | low)
    }

    fn read_utf8_escape(&mut self) -> Result<()> {
        let mut saved = 4;
        self.save_and_next();
        let opened = self.is_current(b'{');
        self.escape_check(opened, "missing '{'")?;
        let mut code = u64::from(self.hex_digit()?);
        loop {
            self.save_and_next();
            if !self.current.is_some_and(|byte| byte.is_ascii_hexdigit()) {
                break;
            }
            saved += 1;
            self.escape_check(code <= MAX_CODE_POINT >> 4, "UTF-8 value too large")?;
            let digit = self.current.map_or(0, hex_value);
            code = (code << 4) + u64::from(digit);
        }
        let closed = self.is_current(b'}');
        self.escape_check(closed, "missing '}'")?;
        self.advance();
        self.remove_saved(saved);
        push_utf8(&mut self.buffer, code);
        Ok(())
    }

    fn read_decimal_escape(&mut self) -> Result<u8> {
        let mut value: u32 = 0;
        let mut count = 0;
        while count < 3 && self.current.is_some_and(|byte| byte.is_ascii_digit()) {
            let digit = self.current.map_or(0, |byte| byte - b'0');
            value = value * 10 + u32::from(digit);
            self.save_and_next();
            count += 1;
        }
        self.escape_check(value <= 255, "decimal escape too large")?;
        self.remove_saved(count);
        Ok(value as u8)
    }

    fn skip_space_escape(&mut self) -> Result<()> {
        self.buffer.pop();
        self.advance();
        while self.current.is_some_and(is_c_space) {
            if is_newline(self.current) {
                self.inc_line()?;
            } else {
                self.advance();
            }
        }
        Ok(())
    }
}

fn simple_escape(byte: u8) -> Option<u8> {
    Some(match byte {
        b'a' => 0x07,
        b'b' => 0x08,
        b'f' => 0x0c,
        b'n' => b'\n',
        b'r' => b'\r',
        b't' => b'\t',
        b'v' => 0x0b,
        b'\\' | b'"' | b'\'' => byte,
        _ => return None,
    })
}

/// Follows `luaO_utf8esc` in lobject.c.
fn push_utf8(buffer: &mut Vec<u8>, mut code: u64) {
    if code < 0x80 {
        buffer.push(code as u8);
        return;
    }
    let mut bytes = [0u8; 8];
    let mut count = 0;
    let mut first_byte_max: u64 = 0x3f;
    loop {
        bytes[count] = (0x80 | (code & 0x3f)) as u8;
        count += 1;
        code >>= 6;
        first_byte_max >>= 1;
        if code <= first_byte_max {
            break;
        }
    }
    bytes[count] = ((!first_byte_max << 1) | code) as u8;
    count += 1;
    buffer.extend(bytes[..count].iter().rev());
}

fn is_newline(byte: Option<u8>) -> bool {
    byte.is_some_and(is_newline_byte)
}

fn is_newline_byte(byte: u8) -> bool {
    matches!(byte, b'\n' | b'\r')
}

fn is_c_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn is_alpha(byte: u8) -> bool {
    byte.is_ascii_alphabetic() || byte == b'_'
}

fn is_alnum(byte: u8) -> bool {
    is_alpha(byte) || byte.is_ascii_digit()
}

#[cfg(test)]
mod tests {
    use super::{Lexer, Sink, Strings, push_utf8};
    use crate::heap::Heap;
    use crate::syntax::token::{Kind, Value};

    #[derive(Default)]
    struct Scan {
        heap: Heap,
        strings: Strings,
    }

    impl Scan {
        fn sink(&mut self) -> Sink<'_> {
            Sink {
                heap: &mut self.heap,
                strings: &mut self.strings,
            }
        }
    }

    fn kinds(source: &str) -> Vec<Kind> {
        let mut scan = Scan::default();
        let mut lexer = Lexer::new(b"t", source.as_bytes(), Vec::new());
        let mut kinds = Vec::new();
        loop {
            let token = lexer.lex(&mut scan.sink()).expect("the source lexes");
            if token.kind == Kind::Eos {
                return kinds;
            }
            kinds.push(token.kind);
        }
    }

    fn first_error(source: &str) -> String {
        let mut scan = Scan::default();
        let mut lexer = Lexer::new(b"t", source.as_bytes(), Vec::new());
        loop {
            match lexer.lex(&mut scan.sink()) {
                Err(error) => return String::from_utf8_lossy(&error).into_owned(),
                Ok(token) if token.kind == Kind::Eos => panic!("no error in {source:?}"),
                Ok(_) => {}
            }
        }
    }

    fn string_value(source: &str) -> Vec<u8> {
        let mut scan = Scan::default();
        let mut lexer = Lexer::new(b"t", source.as_bytes(), Vec::new());
        match lexer.lex(&mut scan.sink()).expect("the string lexes").value {
            Value::Str(string) => scan.heap.bytes(string).to_vec(),
            other => panic!("expected a string, got {other:?}"),
        }
    }

    fn utf8_bytes(code: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_utf8(&mut bytes, code);
        bytes
    }

    #[test]
    fn operators_take_their_longest_form() {
        assert_eq!(
            kinds("a ~= b // c .. d ... e << f >> g <= h >= i == j :: k"),
            vec![
                Kind::Name, Kind::NotEq, Kind::Name, Kind::IntDiv, Kind::Name, Kind::Concat,
                Kind::Name, Kind::Dots, Kind::Name, Kind::Shl, Kind::Name, Kind::Shr, Kind::Name,
                Kind::LessEq, Kind::Name, Kind::GreaterEq, Kind::Name, Kind::EqEq, Kind::Name,
                Kind::DbColon, Kind::Name,
            ]
        );
    }

    #[test]
    fn line_comments_and_long_comments_of_any_level_are_skipped() {
        assert_eq!(kinds("-- short\nx"), vec![Kind::Name]);
        assert_eq!(kinds("--[==[ a ]] ]=] \n ]==] y"), vec![Kind::Name]);
        assert_eq!(kinds("--[ not long\nz"), vec![Kind::Name]);
    }

    #[test]
    fn keywords_are_told_apart_from_names() {
        assert_eq!(
            kinds("local localx _ end1 end"),
            vec![Kind::Local, Kind::Name, Kind::Name, Kind::Name, Kind::End]
        );
    }

    #[test]
    fn long_strings_drop_their_brackets_and_first_newline() {
        assert_eq!(string_value("[[\nab]]"), b"ab".to_vec());
        assert_eq!(string_value("[==[a]]b]==]"), b"a]]b".to_vec());
        assert_eq!(string_value("[[a\r\nb]]"), b"a\nb".to_vec());
    }

    #[test]
    fn equal_long_strings_are_one_string() {
        let mut scan = Scan::default();
        let text = format!("'{0}' '{0}'", "x".repeat(60));
        let mut lexer = Lexer::new(b"t", text.as_bytes(), Vec::new());
        let first = lexer.lex(&mut scan.sink()).expect("lexes").value;
        let second = lexer.lex(&mut scan.sink()).expect("lexes").value;
        assert_eq!(first, second);
    }

    #[test]
    fn escapes_produce_their_bytes() {
        assert_eq!(string_value(r#""\a\b\f\n\r\t\v\\\"\'""#), b"\x07\x08\x0c\n\r\t\x0b\\\"'".to_vec());
        assert_eq!(string_value(r#""\x41\65\0660""#), b"AAB0".to_vec());
        assert_eq!(string_value(r#""\u{48}\u{20AC}""#), "H\u{20ac}".as_bytes().to_vec());
        assert_eq!(string_value("\"a\\z   \n   b\""), b"ab".to_vec());
        assert_eq!(string_value(r#""a\z   \n   b""#), b"a\n   b".to_vec());
        assert_eq!(string_value("\"a\\\nb\""), b"a\nb".to_vec());
    }

    #[test]
    fn utf8_escape_reaches_two_to_the_thirty_first() {
        assert_eq!(utf8_bytes(0x7FFF_FFFF), vec![0xFD, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]);
        assert_eq!(utf8_bytes(0x80), vec![0xC2, 0x80]);
        assert_eq!(string_value(r#""\u{7FFFFFFF}""#), vec![0xFD, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]);
    }

    #[test]
    fn lexer_errors_match_the_reference_text() {
        assert_eq!(first_error("'abc"), "t:1: unfinished string near <eof>");
        assert_eq!(first_error("'ab\ncd'"), "t:1: unfinished string near ''ab'");
        assert_eq!(first_error("'\\q'"), "t:1: invalid escape sequence near ''\\q'");
        assert_eq!(first_error("'\\300'"), "t:1: decimal escape too large near ''\\300''");
        assert_eq!(first_error("'\\x4g'"), "t:1: hexadecimal digit expected near ''\\x4g'");
        assert_eq!(first_error("'\\u41'"), "t:1: missing '{' near ''\\u4'");
        assert_eq!(first_error("[[abc"), "t:1: unfinished long string (starting at line 1) near <eof>");
        assert_eq!(first_error("--[[abc"), "t:1: unfinished long comment (starting at line 1) near <eof>");
        assert_eq!(first_error("[=x"), "t:1: invalid long string delimiter near '[='");
        assert_eq!(first_error("3x"), "t:1: malformed number near '3x'");
        assert_eq!(first_error("3..2"), "t:1: malformed number near '3..2'");
    }

    #[test]
    fn numerals_are_read_with_their_kinds() {
        let mut scan = Scan::default();
        let mut lexer = Lexer::new(b"t", b"0x1p4 9223372036854775808 3. 0xff", Vec::new());
        assert_eq!(lexer.lex(&mut scan.sink()).expect("lexes").value, Value::Float(16.0));
        assert_eq!(lexer.lex(&mut scan.sink()).expect("lexes").value, Value::Float(9_223_372_036_854_775_808.0));
        assert_eq!(lexer.lex(&mut scan.sink()).expect("lexes").value, Value::Float(3.0));
        assert_eq!(lexer.lex(&mut scan.sink()).expect("lexes").value, Value::Int(255));
    }

    #[test]
    fn line_counting_treats_crlf_as_one_break() {
        let mut scan = Scan::default();
        let mut lexer = Lexer::new(b"t", b"a\r\nb\n\rc\n\nd", Vec::new());
        for _ in 0..3 {
            lexer.lex(&mut scan.sink()).expect("lexes");
        }
        assert_eq!(lexer.line, 3);
        lexer.lex(&mut scan.sink()).expect("lexes");
        assert_eq!(lexer.line, 5);
    }
}
