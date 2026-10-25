pub(crate) mod chunk_id;
pub(crate) mod lexer;
pub(crate) mod numeral;
pub(crate) mod token;
#[cfg(test)]
mod tests;

/// Matches PUC's nesting-limit message, which a running message handler sees.
pub(crate) const C_STACK_OVERFLOW: &[u8] = b"C stack overflow";

type Result<T, E = Vec<u8>> = std::result::Result<T, E>;

/// Returns the error `load` would raise for `source`; `chunkname` is shown as `load` shows it, so `=name` and `@file` lose their prefix.
pub fn parse(chunkname: &[u8], source: &[u8]) -> Result<()> {
    let mut heap = crate::heap::Heap::default();
    crate::compiler::compile(&mut heap, chunkname, source, None).map(|_| ())
}
