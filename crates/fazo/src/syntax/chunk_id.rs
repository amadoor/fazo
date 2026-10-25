const ID_SIZE: usize = 60;
const STRING_PREFIX: &[u8] = b"[string \"";
const STRING_SUFFIX: &[u8] = b"\"]";
const ELLIPSIS: &[u8] = b"...";

pub(crate) fn chunk_id(chunkname: &[u8]) -> Vec<u8> {
    match chunkname.first() {
        Some(b'=') => literal_id(&chunkname[1..]),
        Some(b'@') => file_id(&chunkname[1..]),
        _ => string_id(chunkname),
    }
}

fn literal_id(name: &[u8]) -> Vec<u8> {
    let kept = name.len().min(ID_SIZE - 1);
    name[..kept].to_vec()
}

fn file_id(name: &[u8]) -> Vec<u8> {
    if name.len() < ID_SIZE {
        return name.to_vec();
    }
    let kept = ID_SIZE - ELLIPSIS.len() - 1;
    [ELLIPSIS, &name[name.len() - kept..]].concat()
}

fn string_id(source: &[u8]) -> Vec<u8> {
    let room = ID_SIZE - (STRING_PREFIX.len() + ELLIPSIS.len() + STRING_SUFFIX.len()) - 1;
    let newline = source.iter().position(|&byte| byte == b'\n');
    if source.len() < room && newline.is_none() {
        return [STRING_PREFIX, source, STRING_SUFFIX].concat();
    }
    let end = newline.unwrap_or(source.len()).min(room);
    [STRING_PREFIX, &source[..end], ELLIPSIS, STRING_SUFFIX].concat()
}

#[cfg(test)]
mod tests {
    use super::chunk_id;

    #[test]
    fn literal_names_drop_the_equals_sign() {
        assert_eq!(chunk_id(b"=t"), b"t");
        assert_eq!(chunk_id(b"=api.lua"), b"api.lua");
    }

    #[test]
    fn file_names_drop_the_at_sign() {
        assert_eq!(chunk_id(b"@main.lua"), b"main.lua");
        let long = format!("@{}", "d/".repeat(40));
        let id = chunk_id(long.as_bytes());
        assert_eq!(id.len(), 59);
        assert!(id.starts_with(b"..."));
    }

    #[test]
    fn source_strings_are_quoted_and_cut_at_the_first_line() {
        assert_eq!(chunk_id(b"x = = 1"), b"[string \"x = = 1\"]");
        assert_eq!(chunk_id(b"a\nb"), b"[string \"a...\"]");
        let long = "x".repeat(100);
        assert_eq!(chunk_id(long.as_bytes()), format!("[string \"{}...\"]", "x".repeat(45)).as_bytes());
    }

    #[test]
    fn chunk_names_keep_their_bytes() {
        assert_eq!(chunk_id(b"=\xff\xfe"), b"\xff\xfe");
        assert_eq!(chunk_id(b"@\xff.lua"), b"\xff.lua");
        assert_eq!(chunk_id(b"x\xff"), b"[string \"x\xff\"]");
    }
}
