//! `#[serde(with = "crate::bytes")]` stores a byte buffer as one MessagePack `bin` instead of one element per byte.

use std::{fmt, marker::PhantomData, sync::Arc};

use serde::{
    Deserializer, Serializer,
    de::{Error, Visitor},
};

use crate::object::StringBytes;

pub(crate) trait ByteBuffer: Sized {
    fn as_bytes(&self) -> &[u8];

    fn from_bytes(bytes: &[u8]) -> Self;

    fn from_byte_buf(bytes: Vec<u8>) -> Self {
        Self::from_bytes(&bytes)
    }
}

impl ByteBuffer for Vec<u8> {
    fn as_bytes(&self) -> &[u8] {
        self
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        bytes.to_vec()
    }

    fn from_byte_buf(bytes: Vec<u8>) -> Self {
        bytes
    }
}

impl ByteBuffer for StringBytes {
    fn as_bytes(&self) -> &[u8] {
        self
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        StringBytes::from_slice(bytes)
    }
}

impl ByteBuffer for Arc<[u8]> {
    fn as_bytes(&self) -> &[u8] {
        self
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Arc::from(bytes)
    }
}

pub(crate) fn serialize<T: ByteBuffer, S: Serializer>(value: &T, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_bytes(value.as_bytes())
}

pub(crate) fn deserialize<'de, T: ByteBuffer, D: Deserializer<'de>>(deserializer: D) -> Result<T, D::Error> {
    deserializer.deserialize_bytes(BytesVisitor(PhantomData))
}

struct BytesVisitor<T>(PhantomData<T>);

impl<T: ByteBuffer> Visitor<'_> for BytesVisitor<T> {
    type Value = T;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a byte string")
    }

    fn visit_bytes<E: Error>(self, bytes: &[u8]) -> Result<T, E> {
        Ok(T::from_bytes(bytes))
    }

    fn visit_byte_buf<E: Error>(self, bytes: Vec<u8>) -> Result<T, E> {
        Ok(T::from_byte_buf(bytes))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::object::StringBytes;

    #[derive(serde::Serialize, serde::Deserialize, PartialEq, Debug)]
    struct Buffers {
        #[serde(with = "super")]
        vector: Vec<u8>,
        #[serde(with = "super")]
        inline: StringBytes,
        #[serde(with = "super")]
        shared: Arc<[u8]>,
    }

    #[test]
    fn byte_buffers_round_trip_as_messagepack_bin() {
        let buffers = Buffers {
            vector: vec![0, 1, 255],
            inline: StringBytes::from_slice(b"short"),
            shared: Arc::from(&b"shared"[..]),
        };
        let encoded = rmp_serde::to_vec(&buffers).expect("the buffers encode");
        assert_eq!(encoded, [&[0x93, 0xc4, 3, 0, 1, 255, 0xc4, 5][..], b"short", &[0xc4, 6], b"shared"].concat());
        let decoded: Buffers = rmp_serde::from_slice(&encoded).expect("the buffers decode");
        assert_eq!(decoded, buffers);
        assert!(!decoded.inline.spilled(), "a short string stays inline");
    }
}
