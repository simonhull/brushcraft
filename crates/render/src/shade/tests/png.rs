//! A minimal PNG writer and reader for the golden images, ours.
//!
//! 8-bit RGB, no interlace, filter type 0 on every row, the zlib stream in stored
//! (uncompressed) deflate blocks. The files open in any viewer, for a person to look at; the
//! reader reads only what the writer writes.

/// An RGB image, 8 bits a channel, row by row from the top.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Rgb {
    /// Its width.
    pub(crate) width: u32,
    /// Its height.
    pub(crate) height: u32,
    /// Its pixels.
    pub(crate) pixels: Vec<[u8; 3]>,
}

/// The PNG signature.
const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

/// The most bytes a stored deflate block holds.
const STORED: usize = 0xFFFF;

/// The CRC-32 (ISO 3309, the one PNG uses) of `bytes`.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFF_u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// The Adler-32 checksum (zlib's) of `bytes`.
fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1_u32, 0_u32);
    for &byte in bytes {
        a = (a + u32::from(byte)) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// Appends a chunk of `kind` holding `data` to `out`.
fn chunk(out: &mut Vec<u8>, kind: [u8; 4], data: &[u8]) {
    let length = u32::try_from(data.len()).expect("a small chunk");
    out.extend_from_slice(&length.to_be_bytes());
    let start = out.len();
    out.extend_from_slice(&kind);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// `image` as a PNG file.
pub(crate) fn encode(image: &Rgb) -> Vec<u8> {
    let width = usize::try_from(image.width).expect("small");
    let mut raw = Vec::new();
    for row in image.pixels.chunks(width) {
        raw.push(0);
        raw.extend(row.iter().flatten());
    }
    // zlib: deflate with a 32 KiB window, no dictionary, then stored blocks.
    let mut zlib = vec![0x78, 0x01];
    let blocks = raw.chunks(STORED);
    let count = blocks.len();
    for (k, block) in blocks.enumerate() {
        zlib.push(u8::from(k + 1 == count));
        let length = u16::try_from(block.len()).expect("a stored block");
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    zlib.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut header = Vec::new();
    header.extend_from_slice(&image.width.to_be_bytes());
    header.extend_from_slice(&image.height.to_be_bytes());
    // 8 bits, colour type 2 (RGB), deflate, filter method 0, no interlace.
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut out = SIGNATURE.to_vec();
    chunk(&mut out, *b"IHDR", &header);
    chunk(&mut out, *b"IDAT", &zlib);
    chunk(&mut out, *b"IEND", &[]);
    out
}

/// The image in `bytes`, a PNG file [`encode`] wrote; `None` for anything else.
pub(crate) fn decode(bytes: &[u8]) -> Option<Rgb> {
    let mut rest = bytes.strip_prefix(&SIGNATURE)?;
    let mut header = None;
    let mut zlib = Vec::new();
    while !rest.is_empty() {
        let length = usize::try_from(u32::from_be_bytes(rest.get(..4)?.try_into().ok()?)).ok()?;
        let kind = rest.get(4..8)?;
        let data = rest.get(8..8 + length)?;
        let crc = u32::from_be_bytes(rest.get(8 + length..12 + length)?.try_into().ok()?);
        if crc32(rest.get(4..8 + length)?) != crc {
            return None;
        }
        match kind {
            b"IHDR" => header = Some(data.to_vec()),
            b"IDAT" => zlib.extend_from_slice(data),
            _ => {}
        }
        rest = rest.get(12 + length..)?;
    }
    let header = header?;
    let width = u32::from_be_bytes(header.get(..4)?.try_into().ok()?);
    let height = u32::from_be_bytes(header.get(4..8)?.try_into().ok()?);
    if header.get(8..13)? != [8, 2, 0, 0, 0] {
        return None;
    }
    let mut stream = zlib.get(2..)?;
    let mut raw = Vec::new();
    loop {
        let last = *stream.first()? & 1 == 1;
        if *stream.first()? >> 1 != 0 {
            return None;
        }
        let length = usize::from(u16::from_le_bytes(stream.get(1..3)?.try_into().ok()?));
        raw.extend_from_slice(stream.get(5..5 + length)?);
        stream = stream.get(5 + length..)?;
        if last {
            break;
        }
    }
    if u32::from_be_bytes(stream.get(..4)?.try_into().ok()?) != adler32(&raw) {
        return None;
    }
    let row = 1 + 3 * usize::try_from(width).ok()?;
    let mut pixels = Vec::new();
    for line in raw.chunks(row) {
        if line.first() != Some(&0) || line.len() != row {
            return None;
        }
        pixels.extend(line.get(1..)?.as_chunks::<3>().0.iter().copied());
    }
    (pixels.len() == usize::try_from(width * height).ok()?).then_some(Rgb {
        width,
        height,
        pixels,
    })
}

#[test]
fn a_png_round_trips_and_its_checksums_are_the_standard_ones() {
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926, "CRC-32's check value");
    assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398, "Adler-32's example");
    let pixels: Vec<[u8; 3]> = (0..300_u32)
        .map(|k| [0, 7, 13].map(|o| u8::try_from((k * 31 + o) % 251).expect("a byte")))
        .collect();
    let image = Rgb {
        width: 20,
        height: 15,
        pixels,
    };
    let bytes = encode(&image);
    assert_eq!(decode(&bytes), Some(image), "round trip");
    let mut damaged = bytes;
    damaged[60] ^= 1;
    assert_eq!(decode(&damaged), None, "a damaged file is refused");
}
