use super::parse;
use crate::{Options, Value, Vm};

const NAME: &[u8] = b"n";

fn error_of(source: &str) -> String {
    match parse(NAME, source.as_bytes()) {
        Ok(()) => panic!("{source:?} parses but should fail"),
        Err(error) => String::from_utf8_lossy(&error).into_owned(),
    }
}

fn result_of(source: &str) -> String {
    let mut vm = Vm::new(Options::default());
    vm.run_source(b"=t", source.as_bytes())
        .unwrap_or_else(|error| panic!("{source:?}: {}", String::from_utf8_lossy(&vm.error_message(&error))));
    format!("{:?}", vm.global("r"))
}

fn shown(value: Value) -> String {
    format!("{value:?}")
}

#[test]
fn power_is_right_associative_and_binds_tighter_than_unary_minus_on_its_left() {
    assert_eq!(result_of("r = 2^-3^2"), shown(Value::Float(2f64.powi(-9))));
    assert_eq!(result_of("r = -2^2"), shown(Value::Float(-4.0)));
    assert_eq!(result_of("r = 2^3^2"), shown(Value::Float(512.0)));
}

#[test]
fn logical_and_arithmetic_follow_the_priority_table() {
    assert_eq!(result_of("r = false or true and nil"), shown(Value::Nil));
    assert_eq!(result_of("r = 1 + 2 * 3 - 4"), shown(Value::Integer(3)));
    assert_eq!(result_of("r = 1 | 2 & 3 ~ 4"), shown(Value::Integer(7)));
    assert_eq!(result_of("r = 1 << 2 + 1"), shown(Value::Integer(8)));
    assert_eq!(result_of("r = 1 + 2 .. 3 == '33'"), shown(Value::Boolean(true)));
    assert_eq!(result_of("r = not nil == true"), shown(Value::Boolean(true)));
}

#[test]
fn accepted_programs_parse() {
    let sources = [
        "local a <const>, b <close> = 1, nil",
        "goto done; do ::again:: end ::done::",
        "for i = 1, 3, 1 do break end",
        "for k, v in pairs(t) do end",
        "repeat local z = 1 until z",
        "function a.b.c:d(x, ...) return ... end",
        "local t = {1, 2; x = 3, [4] = 5,}",
        "x = a.b[c]:d 'str' {1}",
        "while x do if x then break else y = 1 end end",
        "while true do goto out end ::out::",
        "global x, y = 1, 2; global function g() end; global * ",
        "local function f(...t) return t[1], t.n end",
        "return",
    ];
    for source in sources {
        parse(NAME, source.as_bytes())
            .unwrap_or_else(|error| panic!("{source:?}: {}", String::from_utf8_lossy(&error)));
    }
}

#[test]
fn syntax_errors_match_the_reference_text() {
    let cases: &[(&str, &str)] = &[
        ("x = = 1", "[string \"n\"]:1: unexpected symbol near '='"),
        ("for i 1 do end", "[string \"n\"]:1: '=' or 'in' expected near '1'"),
        ("if x then\nprint(1)", "[string \"n\"]:2: 'end' expected (to close 'if' at line 1) near <eof>"),
        ("x = \"abc", "[string \"n\"]:1: unfinished string near <eof>"),
        ("x = 3x", "[string \"n\"]:1: malformed number near '3x'"),
        ("local x <const> = 1; x = 2", "[string \"n\"]:1: attempt to assign to const variable 'x'"),
        ("break", "[string \"n\"]:1: break outside loop near 'break'"),
        ("if x then break end", "[string \"n\"]:1: break outside loop near 'break'"),
        ("goto nowhere", "[string \"n\"]:1: no visible label 'nowhere' for <goto> at line 1"),
        ("goto f; local x = 1; ::f:: print(x)", "[string \"n\"]:1: <goto f> at line 1 jumps into the scope of 'x'"),
        ("::a:: ::a::", "[string \"n\"]:1: label 'a' already defined on line 1"),
        ("::l:: do ::l:: end", "[string \"n\"]:1: label 'l' already defined on line 1"),
        ("function f() return ... end", "[string \"n\"]:1: cannot use '...' outside a vararg function near '...'"),
        ("local a <close>, b <close> = 1, 2", "[string \"n\"]:1: multiple to-be-closed variables in local list"),
        ("local a <foo> = 1", "[string \"n\"]:1: unknown attribute 'foo'"),
        ("x", "[string \"n\"]:1: syntax error near <eof>"),
        ("end", "[string \"n\"]:1: <eof> expected near 'end'"),
        ("x = f 1", "[string \"n\"]:1: unexpected symbol near '1'"),
        ("local = 1", "[string \"n\"]:1: <name> expected near '='"),
        ("x = ", "[string \"n\"]:1: unexpected symbol near <eof>"),
        ("x = [=", "[string \"n\"]:1: invalid long string delimiter near '[='"),
        ("x = [[abc", "[string \"n\"]:1: unfinished long string (starting at line 1) near <eof>"),
        ("x = 1 --[==[ never closed", "[string \"n\"]:1: unfinished long comment (starting at line 1) near <eof>"),
        (r#"x = "\300""#, "[string \"n\"]:1: decimal escape too large near '\"\\300\"'"),
        (r#"x = "\q""#, "[string \"n\"]:1: invalid escape sequence near '\"\\q'"),
        (r#"x = "\xZZ""#, "[string \"n\"]:1: hexadecimal digit expected near '\"\\xZ'"),
        (r#"x = "\u{80000000}""#, "[string \"n\"]:1: UTF-8 value too large near '\"\\u{80000000'"),
        ("f(1 2)", "[string \"n\"]:1: ')' expected near '2'"),
        ("x = {1 2}", "[string \"n\"]:1: '}' expected near '2'"),
        ("function f()", "[string \"n\"]:1: 'end' expected near <eof>"),
        ("local t = {\n1,\n2", "[string \"n\"]:3: '}' expected (to close '{' at line 1) near <eof>"),
        ("f(\n1,\n", "[string \"n\"]:3: unexpected symbol near <eof>"),
        ("for i = 1, 2 do\n  x = 1\n", "[string \"n\"]:3: 'end' expected (to close 'for' at line 1) near <eof>"),
        ("x = 1\nreturn 1 y", "[string \"n\"]:2: <eof> expected near 'y'"),
        ("local x <const> = 1; local function g() x = 2 end", "[string \"n\"]:1: attempt to assign to const variable 'x'"),
        ("local z <close> = nil; z = 1", "[string \"n\"]:1: attempt to assign to const variable 'z'"),
        ("function g(..., a) end", "[string \"n\"]:1: ')' expected near ','"),
        ("local a <const>, b = 1, 2; a = 3", "[string \"n\"]:1: attempt to assign to const variable 'a'"),
        ("(a) = 1", "[string \"n\"]:1: syntax error near '='"),
        ("f() = 1", "[string \"n\"]:1: syntax error near '='"),
        ("a, f() = 1", "[string \"n\"]:1: syntax error near '='"),
        ("a.b:c = 1", "[string \"n\"]:1: function arguments expected near '='"),
        ("global x; y = 1", "[string \"n\"]:1: variable 'y' not declared"),
        ("global <const> x; x = 1", "[string \"n\"]:1: attempt to assign to const variable 'x'"),
        ("global <close> x", "[string \"n\"]:1: global variables cannot be to-be-closed"),
        ("local function f(...t) t = 1 end", "[string \"n\"]:1: attempt to assign to const variable 't'"),
    ];
    for (snippet, expected) in cases {
        assert_eq!(error_of(snippet), *expected, "snippet {snippet:?}");
    }
}

#[test]
fn limit_errors_name_the_limit_and_the_function() {
    let locals = format!("local {} = 1", (0..201).map(|index| format!("a{index}")).collect::<Vec<_>>().join(","));
    assert_eq!(
        error_of(&format!("{locals}\n")),
        "[string \"n\"]:2: too many local variables (limit is 200) in main function near <eof>"
    );
    let values = format!("local x = {{}}\nreturn {}\n", vec!["1"; 300].join(","));
    assert_eq!(
        error_of(&values),
        "[string \"n\"]:2: too many registers (limit is 255) in main function near '1'"
    );
}

#[test]
fn unicode_and_long_escape_limits_are_checked_as_the_reference_does() {
    assert_eq!(
        error_of(r#"x = "\u{}""#),
        "[string \"n\"]:1: hexadecimal digit expected near '\"\\u{}'"
    );
}
