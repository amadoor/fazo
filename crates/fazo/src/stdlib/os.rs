
use super::{arg_error, argument, check_integer, exec_result, failure_values, got};
use crate::{
    LuaError, Values, values, Vm,
    host::{HostError, TimeFields},
    number,
    object::NativeFn,
    value::Value,
};

pub(super) fn open(vm: &mut Vm) {
    let functions: &[(&'static str, NativeFn)] = &[
        ("clock", clock),
        ("date", date),
        ("difftime", difftime),
        ("getenv", getenv),
        ("setlocale", setlocale),
        ("execute", execute),
        ("exit", exit),
        ("remove", remove),
        ("rename", rename),
        ("tmpname", tmpname),
        ("time", time),
    ];
    let table = vm.new_table();
    for (name, function) in functions {
        let value = vm.new_native(name, *function);
        vm.set_field(table, name, value);
    }
    vm.set_global("os", Value::Table(table));
}

fn clock(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    Ok(values![Value::Float(vm.host.system.processor_seconds())])
}

fn difftime(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let first = time_argument(vm, args, 0, "difftime")?;
    let second = time_argument(vm, args, 1, "difftime")?;
    Ok(values![Value::Float((first - second) as f64)])
}

fn time_argument(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<i64, LuaError> {
    check_integer(vm, args, index, name)
}

fn getenv(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = match argument(args, 0) {
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => {
            let message = format!("string expected, got {}", got(vm, args, 0));
            return Err(arg_error(vm, 1, "getenv", &message));
        }
    };
    match vm.host.system.getenv(&name) {
        Some(value) => Ok(values![vm.string_value(&value)]),
        None => Ok(values![Value::Nil]),
    }
}

fn setlocale(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    const CATEGORIES: [&[u8]; 6] = [b"all", b"collate", b"ctype", b"monetary", b"numeric", b"time"];
    let locale = match argument(args, 0) {
        Value::Nil => None,
        Value::String(string) => Some(vm.heap_bytes(string).to_vec()),
        _ => return Err(arg_error(vm, 1, "setlocale", "string expected")),
    };
    let category = match argument(args, 1) {
        Value::Nil => b"all".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 2, "setlocale", "string expected")),
    };
    if !CATEGORIES.contains(&category.as_slice()) {
        let name = String::from_utf8_lossy(&category).into_owned();
        return Err(arg_error(vm, 2, "setlocale", &format!("invalid option '{name}'")));
    }
    match locale.as_deref() {
        None | Some(b"" | b"C" | b"POSIX") => Ok(values![vm.string_value(b"C")]),
        Some(_) => Ok(values![Value::Nil]),
    }
}

fn execute(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let command = match argument(args, 0) {
        Value::Nil => {
            let available = vm.host.system.shell_available();
            return Ok(values![Value::Boolean(available)]);
        }
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        value @ (Value::Integer(_) | Value::Float(_)) => vm.to_display_bytes(value),
        _ => {
            let message = format!("string expected, got {}", got(vm, args, 0));
            return Err(arg_error(vm, 1, "execute", &message));
        }
    };
    match vm.host.system.execute(&command) {
        Ok(status) => Ok(exec_result(vm, status)),
        Err(error) => Ok(failure_values(vm, &error, None)),
    }
}

fn exit(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let code = match argument(args, 0) {
        Value::Nil | Value::Boolean(true) => 0,
        Value::Boolean(false) => 1,
        _ => check_integer(vm, args, 0, "exit")? as i32,
    };
    if !argument(args, 1).is_falsy() {
        vm.close();
    }
    vm.host.system.exit(code);
    Err(vm.runtime_error_text("os.exit is not available on this host"))
}

fn file_name_argument(vm: &mut Vm, args: &[Value], index: usize, name: &str) -> Result<Vec<u8>, LuaError> {
    match argument(args, index) {
        Value::String(string) => Ok(vm.heap_bytes(string).to_vec()),
        value @ (Value::Integer(_) | Value::Float(_)) => Ok(vm.to_display_bytes(value)),
        _ => {
            let got = args.get(index).map_or("no value", |value| value.type_name());
            Err(arg_error(vm, index + 1, name, &format!("string expected, got {got}")))
        }
    }
}

fn remove(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let name = file_name_argument(vm, args, 0, "remove")?;
    match vm.host.files.remove(&name) {
        Ok(()) => Ok(values![Value::Boolean(true)]),
        Err(error) => Ok(failure_values(vm, &error, Some(&name))),
    }
}

fn rename(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let from = file_name_argument(vm, args, 0, "rename")?;
    let to = file_name_argument(vm, args, 1, "rename")?;
    match vm.host.files.rename(&from, &to) {
        Ok(()) => Ok(values![Value::Boolean(true)]),
        Err(error) => Ok(failure_values(vm, &error, None)),
    }
}

fn tmpname(vm: &mut Vm, _args: &[Value]) -> Result<Values, LuaError> {
    match vm.host.files.tmpname() {
        Ok(name) => Ok(values![vm.string_value(&name)]),
        Err(HostError { reason, .. }) => Err(vm.runtime_error_text(&format!("unable to generate a unique filename: {reason}"))),
    }
}

fn now(vm: &mut Vm) -> i64 {
    vm.host.system.now()
}

fn time(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let Value::Table(table) = argument(args, 0) else {
        if argument(args, 0).is_nil() {
            return Ok(values![Value::Integer(now(vm))]);
        }
        let message = format!("table expected, got {}", got(vm, args, 0));
        return Err(arg_error(vm, 1, "time", &message));
    };
    let year = date_field(vm, table, "year", None, 1900)?;
    let month = date_field(vm, table, "month", None, 1)?;
    let day = date_field(vm, table, "day", None, 0)?;
    let hour = date_field(vm, table, "hour", Some(12), 0)?;
    let minute = date_field(vm, table, "min", Some(0), 0)?;
    let second = date_field(vm, table, "sec", Some(0), 0)?;
    let local = days_of(year, month, day) * 86_400 + hour * 3600 + minute * 60 + second;
    let offset = local_offset(vm, local);
    let seconds = local - offset;
    if let Some(normalized) = vm.host.system.break_down(seconds, false) {
        store_fields(vm, table, normalized);
    }
    Ok(values![Value::Integer(seconds)])
}

fn date_field(
    vm: &mut Vm,
    table: crate::arena::Ref<crate::table::Table>,
    name: &str,
    default: Option<i64>,
    delta: i64,
) -> Result<i64, LuaError> {
    let field = vm.raw_field(table, name);
    if field.is_nil() {
        return match default {
            Some(value) => Ok(value),
            None => Err(vm.runtime_error_text(&format!("field '{name}' missing in date table"))),
        };
    }
    let integer = vm
        .coerce_number(field)
        .and_then(|number| number::to_integer(number, number::Rounding::Exact));
    let Some(value) = integer else {
        return Err(vm.runtime_error_text(&format!("field '{name}' is not an integer")));
    };
    let fits = if value >= 0 {
        value - delta <= i64::from(i32::MAX)
    } else {
        i64::from(i32::MIN) + delta <= value
    };
    if !fits {
        return Err(vm.runtime_error_text(&format!("field '{name}' is out-of-bound")));
    }
    Ok(value)
}

fn local_offset(vm: &mut Vm, seconds: i64) -> i64 {
    let Some(local) = vm.host.system.break_down(seconds, false) else {
        return 0;
    };
    let Some(utc) = vm.host.system.break_down(seconds, true) else {
        return 0;
    };
    let as_seconds = |fields: TimeFields| {
        days_of(fields.year, fields.month, fields.day) * 86_400
            + fields.hour * 3600
            + fields.minute * 60
            + fields.second
    };
    as_seconds(local) - as_seconds(utc)
}

fn days_of(year: i64, month: i64, day: i64) -> i64 {
    let total = year * 12 + (month - 1);
    let year = total.div_euclid(12);
    let month = total.rem_euclid(12) + 1;
    let shifted_year = if month <= 2 { year - 1 } else { year };
    let era = shifted_year.div_euclid(400);
    let year_of_era = shifted_year.rem_euclid(400);
    let month_index = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * month_index + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn store_fields(vm: &mut Vm, table: crate::arena::Ref<crate::table::Table>, fields: TimeFields) {
    let entries = [
        ("year", fields.year),
        ("month", fields.month),
        ("day", fields.day),
        ("hour", fields.hour),
        ("min", fields.minute),
        ("sec", fields.second),
        ("yday", fields.yearday),
        ("wday", fields.weekday),
    ];
    for (name, value) in entries {
        vm.set_field(table, name, Value::Integer(value));
    }
    if let Some(saving) = fields.daylight_saving {
        vm.set_field(table, "isdst", Value::Boolean(saving));
    }
}

fn date(vm: &mut Vm, args: &[Value]) -> Result<Values, LuaError> {
    let format = match argument(args, 0) {
        Value::Nil => b"%c".to_vec(),
        Value::String(string) => vm.heap_bytes(string).to_vec(),
        _ => return Err(arg_error(vm, 1, "date", "string expected")),
    };
    let seconds = if argument(args, 1).is_nil() {
        now(vm)
    } else {
        check_integer(vm, args, 1, "date")?
    };
    let (utc, body) = match format.strip_prefix(b"!") {
        Some(rest) => (true, rest.to_vec()),
        None => (false, format),
    };
    let Some(fields) = vm.host.system.break_down(seconds, utc) else {
        return Err(vm.runtime_error_text("date result cannot be represented in this installation"));
    };
    if body == b"*t" {
        let table = vm.new_table();
        store_fields(vm, table, fields);
        return Ok(values![Value::Table(table)]);
    }
    let text = format_date(vm, &body, fields)?;
    Ok(values![vm.string_value(&text)])
}

const WEEKDAYS: [&str; 7] = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS: [&str; 12] = [
    "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November",
    "December",
];

#[cfg(windows)]
const SINGLE_CONVERSIONS: &[u8] = b"aAbBcdHIjmMpSUwWxXyYzZ%";
#[cfg(windows)]
const DOUBLE_CONVERSIONS: &[&[u8]] = &[
    b"#c", b"#x", b"#d", b"#H", b"#I", b"#j", b"#m", b"#M", b"#S", b"#U", b"#w", b"#W", b"#y", b"#Y",
];
#[cfg(not(windows))]
const SINGLE_CONVERSIONS: &[u8] = b"aAbBcCdDeFgGhHIjmMnprRStTuUVwWxXyYzZ%";
#[cfg(not(windows))]
const DOUBLE_CONVERSIONS: &[&[u8]] = &[
    b"Ec", b"EC", b"Ex", b"EX", b"Ey", b"EY", b"Od", b"Oe", b"OH", b"OI", b"Om", b"OM", b"OS", b"Ou", b"OU",
    b"OV", b"Ow", b"OW", b"Oy",
];

fn format_date(vm: &mut Vm, format: &[u8], fields: TimeFields) -> Result<Vec<u8>, LuaError> {
    let mut out = Vec::new();
    let mut rest = format;
    while let Some((&byte, after)) = rest.split_first() {
        if byte != b'%' {
            out.push(byte);
            rest = after;
            continue;
        }
        let length = conversion_length(vm, after)?;
        rest = &after[length..];
        let text = convert(&after[..length], fields);
        out.extend_from_slice(text.as_bytes());
    }
    Ok(out)
}

fn conversion_length(vm: &mut Vm, after: &[u8]) -> Result<usize, LuaError> {
    if after.first().is_some_and(|byte| SINGLE_CONVERSIONS.contains(byte)) {
        return Ok(1);
    }
    if DOUBLE_CONVERSIONS.iter().any(|option| after.starts_with(option)) {
        return Ok(2);
    }
    let text = after.split(|&byte| byte == 0).next().unwrap_or_default();
    let message = format!("invalid conversion specifier '%{}'", String::from_utf8_lossy(text));
    Err(arg_error(vm, 1, "date", &message))
}

fn convert(conversion: &[u8], fields: TimeFields) -> String {
    match conversion {
        [b'#', letter] => alternate(*letter, fields),
        [b'E' | b'O', letter] => plain(*letter, fields),
        [letter] => plain(*letter, fields),
        _ => String::new(),
    }
}

fn alternate(letter: u8, fields: TimeFields) -> String {
    let weekday = WEEKDAYS[(fields.weekday - 1) as usize % 7];
    let month = MONTHS[(fields.month - 1) as usize % 12];
    match letter {
        b'c' => format!(
            "{weekday}, {month} {:02}, {} {:02}:{:02}:{:02}",
            fields.day, fields.year, fields.hour, fields.minute, fields.second
        ),
        b'x' => format!("{weekday}, {month} {:02}, {}", fields.day, fields.year),
        _ => {
            let number = plain(letter, fields);
            match number.trim_start_matches('0') {
                "" => "0".to_string(),
                trimmed => trimmed.to_string(),
            }
        }
    }
}

fn plain(conversion: u8, fields: TimeFields) -> String {
    let weekday = WEEKDAYS[(fields.weekday - 1) as usize % 7];
    let month = MONTHS[(fields.month - 1) as usize % 12];
    let hour12 = match fields.hour % 12 {
        0 => 12,
        hour => hour,
    };
    match conversion {
        b'a' => weekday[..3].to_string(),
        b'A' => weekday.to_string(),
        b'b' | b'h' => month[..3].to_string(),
        b'B' => month.to_string(),
        b'c' => format!(
            "{} {} {:>2} {:02}:{:02}:{:02} {}",
            &weekday[..3],
            &month[..3],
            fields.day,
            fields.hour,
            fields.minute,
            fields.second,
            fields.year
        ),
        b'C' => format!("{:02}", fields.year / 100),
        b'd' => format!("{:02}", fields.day),
        b'D' | b'x' => format!("{:02}/{:02}/{:02}", fields.month, fields.day, fields.year % 100),
        b'e' => format!("{:>2}", fields.day),
        b'F' => format!("{}-{:02}-{:02}", fields.year, fields.month, fields.day),
        b'H' => format!("{:02}", fields.hour),
        b'I' => format!("{hour12:02}"),
        b'j' => format!("{:03}", fields.yearday),
        b'm' => format!("{:02}", fields.month),
        b'M' => format!("{:02}", fields.minute),
        b'n' => "\n".to_string(),
        b'p' => (if fields.hour < 12 { "AM" } else { "PM" }).to_string(),
        b'r' => format!(
            "{:02}:{:02}:{:02} {}",
            hour12,
            fields.minute,
            fields.second,
            if fields.hour < 12 { "AM" } else { "PM" }
        ),
        b'R' => format!("{:02}:{:02}", fields.hour, fields.minute),
        b'S' => format!("{:02}", fields.second),
        b't' => "\t".to_string(),
        b'T' | b'X' => format!("{:02}:{:02}:{:02}", fields.hour, fields.minute, fields.second),
        b'u' => format!("{}", (fields.weekday + 5) % 7 + 1),
        b'w' => format!("{}", fields.weekday - 1),
        b'y' => format!("{:02}", fields.year % 100),
        b'Y' => format!("{}", fields.year),
        b'g' => format!("{:02}", iso_week(fields).0.rem_euclid(100)),
        b'G' => format!("{}", iso_week(fields).0),
        b'V' => format!("{:02}", iso_week(fields).1),
        b'U' => format!("{:02}", (fields.yearday - 1 + 7 - (fields.weekday - 1)) / 7),
        b'W' => format!("{:02}", (fields.yearday - 1 + 7 - (fields.weekday + 5) % 7) / 7),
        b'%' => "%".to_string(),
        b'z' => "+0000".to_string(),
        b'Z' => "UTC".to_string(),
        _ => String::new(),
    }
}

fn iso_week(fields: TimeFields) -> (i64, i64) {
    let weekday = (fields.weekday + 5) % 7 + 1;
    let week = (fields.yearday - weekday + 10) / 7;
    if week < 1 {
        return (fields.year - 1, weeks_in_year(fields.year - 1));
    }
    if week > weeks_in_year(fields.year) {
        return (fields.year + 1, 1);
    }
    (fields.year, week)
}

fn weeks_in_year(year: i64) -> i64 {
    let p = |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
    if p(year) == 4 || p(year - 1) == 3 { 53 } else { 52 }
}

#[cfg(test)]
mod tests {
    use super::days_of;

    #[test]
    fn days_are_counted_from_the_epoch_with_month_overflow() {
        assert_eq!(days_of(1970, 1, 1), 0);
        assert_eq!(days_of(1970, 13, 1), days_of(1971, 1, 1));
        assert_eq!(days_of(2000, 3, 1) - days_of(2000, 2, 1), 29);
    }
}
