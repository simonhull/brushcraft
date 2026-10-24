//! Reading back the JSON lines the harness writes: one flat object a line.
//!
//! [`crate::report`] writes records by hand, and this reads them, for `perf` to take the
//! subcommands' results and its own history in. It reads what the writer writes and a little
//! more (`true`, `false`, nested arrays, any whitespace between tokens), and refuses the rest
//! with the byte offset where it stopped: nested objects, trailing text, malformed numbers or
//! escapes.

use thiserror::Error;

/// A JSON value inside a flat object.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Json {
    /// `null`: what the writer puts for a float that isn't finite.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number. Every number the harness writes fits an `f64` exactly or near enough: counts
    /// stay far below 2^53.
    Number(f64),
    /// A string, escapes resolved.
    String(String),
    /// An array of values.
    Array(Vec<Self>),
}

impl Json {
    /// The value as a number; `None` if it is anything else.
    pub(crate) const fn as_number(&self) -> Option<f64> {
        match self {
            Self::Number(x) => Some(*x),
            Self::Null | Self::Bool(_) | Self::String(_) | Self::Array(_) => None,
        }
    }

    /// The value as a string; `None` if it is anything else.
    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s),
            Self::Null | Self::Bool(_) | Self::Number(_) | Self::Array(_) => None,
        }
    }

    /// The value as a boolean; `None` if it is anything else.
    pub(crate) const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            Self::Null | Self::Number(_) | Self::String(_) | Self::Array(_) => None,
        }
    }
}

/// Why a line isn't a flat JSON object.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum JsonError {
    /// The text ended inside a value.
    #[error("the line ends early")]
    End,
    /// A character that can't start or continue what was expected here.
    #[error("unexpected {found:?} at byte {at}")]
    Unexpected {
        /// The character found.
        found: char,
        /// Its byte offset in the line.
        at: usize,
    },
    /// A number that doesn't parse, or isn't finite.
    #[error("bad number {text:?} at byte {at}")]
    Number {
        /// The number's text.
        text: String,
        /// Its byte offset in the line.
        at: usize,
    },
    /// A `\u` escape that isn't four hex digits naming a character (surrogates included: the
    /// writer never escapes past `\u001F`).
    #[error("bad escape at byte {at}")]
    Escape {
        /// The escape's byte offset in the line.
        at: usize,
    },
    /// An object inside the object: the harness's records are flat.
    #[error("nested object at byte {at}")]
    Nested {
        /// The inner object's byte offset in the line.
        at: usize,
    },
}

/// A flat JSON object: its fields in the order they appear.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct Object {
    /// The fields; a name given twice keeps both, and [`Object::get`] finds the first.
    fields: Vec<(String, Json)>,
}

impl Object {
    /// Parses `line` (surrounding whitespace allowed) as one flat object.
    ///
    /// # Errors
    ///
    /// If it isn't one: see [`JsonError`].
    pub(crate) fn parse(line: &str) -> Result<Self, JsonError> {
        let mut reader = Reader { text: line, at: 0 };
        let object = reader.object()?;
        reader.skip_space();
        match reader.peek() {
            None => Ok(object),
            Some(found) => Err(JsonError::Unexpected {
                found,
                at: reader.at,
            }),
        }
    }

    /// The value of the field `name`, if there is one.
    pub(crate) fn get(&self, name: &str) -> Option<&Json> {
        self.fields
            .iter()
            .find(|(field, _)| field == name)
            .map(|(_, value)| value)
    }

    /// The fields, in order.
    #[cfg(test)]
    pub(crate) fn fields(&self) -> &[(String, Json)] {
        &self.fields
    }
}

/// A cursor over a line of JSON.
#[derive(Debug)]
struct Reader<'a> {
    /// The line.
    text: &'a str,
    /// The byte offset of the next character.
    at: usize,
}

impl Reader<'_> {
    /// The next character, without taking it.
    fn peek(&self) -> Option<char> {
        self.text.get(self.at..)?.chars().next()
    }

    /// Takes the next character.
    ///
    /// # Errors
    ///
    /// At the end of the line.
    fn next(&mut self) -> Result<char, JsonError> {
        let c = self.peek().ok_or(JsonError::End)?;
        self.at += c.len_utf8();
        Ok(c)
    }

    /// Skips JSON whitespace.
    fn skip_space(&mut self) {
        while self
            .peek()
            .is_some_and(|c| matches!(c, ' ' | '\t' | '\n' | '\r'))
        {
            self.at += 1;
        }
    }

    /// Takes `want` after any whitespace.
    ///
    /// # Errors
    ///
    /// If the next character is another, or there is none.
    fn expect(&mut self, want: char) -> Result<(), JsonError> {
        self.skip_space();
        let at = self.at;
        match self.next()? {
            c if c == want => Ok(()),
            found => Err(JsonError::Unexpected { found, at }),
        }
    }

    /// An object of name–value pairs.
    ///
    /// # Errors
    ///
    /// As [`Object::parse`].
    fn object(&mut self) -> Result<Object, JsonError> {
        self.expect('{')?;
        let mut fields = Vec::new();
        self.skip_space();
        if self.peek() == Some('}') {
            self.at += 1;
            return Ok(Object { fields });
        }
        loop {
            self.skip_space();
            let name = self.string()?;
            self.expect(':')?;
            let value = self.value()?;
            fields.push((name, value));
            self.skip_space();
            let at = self.at;
            match self.next()? {
                ',' => {}
                '}' => return Ok(Object { fields }),
                found => return Err(JsonError::Unexpected { found, at }),
            }
        }
    }

    /// Any value but an object.
    ///
    /// # Errors
    ///
    /// As [`Object::parse`].
    fn value(&mut self) -> Result<Json, JsonError> {
        self.skip_space();
        let at = self.at;
        match self.peek().ok_or(JsonError::End)? {
            '"' => Ok(Json::String(self.string()?)),
            '[' => self.array(),
            '{' => Err(JsonError::Nested { at }),
            't' => self.word("true", Json::Bool(true)),
            'f' => self.word("false", Json::Bool(false)),
            'n' => self.word("null", Json::Null),
            '-' | '0'..='9' => self.number(),
            found => Err(JsonError::Unexpected { found, at }),
        }
    }

    /// The literal `word`, which stands for `value`.
    ///
    /// # Errors
    ///
    /// If the text there isn't `word`.
    fn word(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        let at = self.at;
        if self
            .text
            .get(at..)
            .is_some_and(|rest| rest.starts_with(word))
        {
            self.at += word.len();
            Ok(value)
        } else {
            Err(JsonError::Unexpected {
                found: self.peek().unwrap_or(' '),
                at,
            })
        }
    }

    /// An array of values.
    ///
    /// # Errors
    ///
    /// As [`Object::parse`].
    fn array(&mut self) -> Result<Json, JsonError> {
        self.expect('[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(']') {
            self.at += 1;
            return Ok(Json::Array(items));
        }
        loop {
            items.push(self.value()?);
            self.skip_space();
            let at = self.at;
            match self.next()? {
                ',' => {}
                ']' => return Ok(Json::Array(items)),
                found => return Err(JsonError::Unexpected { found, at }),
            }
        }
    }

    /// A number, in JSON's grammar: an optional minus, digits, an optional fraction and
    /// exponent.
    ///
    /// # Errors
    ///
    /// If the text doesn't follow the grammar, or names a number past `f64`'s range.
    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
        {
            self.at += 1;
        }
        let text = self.text.get(start..self.at).unwrap_or_default();
        let bad = || JsonError::Number {
            text: text.to_owned(),
            at: start,
        };
        if !follows_grammar(text) {
            return Err(bad());
        }
        let x: f64 = text
            .parse()
            .map_err(|_grammar_checked: std::num::ParseFloatError| bad())?;
        if x.is_finite() {
            Ok(Json::Number(x))
        } else {
            Err(bad())
        }
    }

    /// A string, its escapes resolved.
    ///
    /// # Errors
    ///
    /// If it doesn't start with a quote, ends early, holds a raw control character, or an
    /// escape is malformed.
    fn string(&mut self) -> Result<String, JsonError> {
        self.expect('"')?;
        let mut out = String::new();
        loop {
            let at = self.at;
            match self.next()? {
                '"' => return Ok(out),
                '\\' => out.push(self.escape(at)?),
                c if u32::from(c) < 0x20 => return Err(JsonError::Unexpected { found: c, at }),
                c => out.push(c),
            }
        }
    }

    /// The character an escape stands for, its backslash (at byte `at`) already taken.
    ///
    /// # Errors
    ///
    /// If the escape is unknown, or a `\u` isn't four hex digits naming a character.
    fn escape(&mut self, at: usize) -> Result<char, JsonError> {
        Ok(match self.next()? {
            '"' => '"',
            '\\' => '\\',
            '/' => '/',
            'b' => '\u{8}',
            'f' => '\u{C}',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'u' => {
                let digits = self
                    .text
                    .get(self.at..self.at + 4)
                    .ok_or(JsonError::Escape { at })?;
                let code =
                    u32::from_str_radix(digits, 16).map_err(|_not_hex| JsonError::Escape { at })?;
                self.at += 4;
                char::from_u32(code).ok_or(JsonError::Escape { at })?
            }
            _ => return Err(JsonError::Escape { at }),
        })
    }
}

/// Whether `text` is a number in JSON's grammar: `-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`.
///
/// Rust's float parser takes more (`+1`, `.5`, `1.`, `inf`), so the grammar is checked first.
fn follows_grammar(text: &str) -> bool {
    /// `s` with its leading ASCII digits taken off, and how many there were.
    fn digits(s: &str) -> (&str, usize) {
        let n = s.bytes().take_while(u8::is_ascii_digit).count();
        (s.get(n..).unwrap_or_default(), n)
    }
    let unsigned = text.strip_prefix('-').unwrap_or(text);
    let (rest, int) = digits(unsigned);
    // No leading zeros: `0` alone, or digits not starting with one.
    if int == 0 || (int > 1 && unsigned.starts_with('0')) {
        return false;
    }
    let after_fraction = match rest.strip_prefix('.') {
        Some(fraction) => match digits(fraction) {
            (_, 0) => return false,
            (after, _) => after,
        },
        None => rest,
    };
    let after_exponent = match after_fraction.strip_prefix(['e', 'E']) {
        Some(exponent) => match digits(exponent.strip_prefix(['+', '-']).unwrap_or(exponent)) {
            (_, 0) => return false,
            (after, _) => after,
        },
        None => after_fraction,
    };
    after_exponent.is_empty()
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::report::Record;

    /// Whatever the writer writes, the reader reads back, field by field.
    #[test]
    fn records_read_back() {
        let line = Record::new("brush", "stroke")
            .with("name", "a \"b\"\\\n\t\u{1}")
            .with("n", 3_u64)
            .with("x", -1.5e-7)
            .with("bad", f64::NAN)
            .with("xs", vec![1.0, 2.25])
            .with("empty", Vec::<f64>::new())
            .to_json();
        let object = Object::parse(&line).unwrap();
        assert_eq!(
            object.fields(),
            [
                ("bench".to_owned(), Json::String("brush".to_owned())),
                ("result".to_owned(), Json::String("stroke".to_owned())),
                (
                    "name".to_owned(),
                    Json::String("a \"b\"\\\n\t\u{1}".to_owned())
                ),
                ("n".to_owned(), Json::Number(3.0)),
                ("x".to_owned(), Json::Number(-1.5e-7)),
                ("bad".to_owned(), Json::Null),
                (
                    "xs".to_owned(),
                    Json::Array(vec![Json::Number(1.0), Json::Number(2.25)])
                ),
                ("empty".to_owned(), Json::Array(Vec::new())),
            ],
            "{line}"
        );
        assert_eq!(
            object.get("n").and_then(Json::as_number),
            Some(3.0),
            "a number"
        );
        assert_eq!(
            object.get("bench").and_then(Json::as_str),
            Some("brush"),
            "a string"
        );
        assert_eq!(object.get("missing"), None, "no such field");
    }

    #[test]
    fn spacing_booleans_and_escapes_are_read() {
        let object =
            Object::parse(" { \"a\" : true ,\"b\":false, \"c\": [ [1], \"\\u00e9\\/\" ] } ")
                .unwrap();
        assert_eq!(object.get("a").and_then(Json::as_bool), Some(true), "true");
        assert_eq!(
            object.get("b").and_then(Json::as_bool),
            Some(false),
            "false"
        );
        assert_eq!(
            object.get("c"),
            Some(&Json::Array(vec![
                Json::Array(vec![Json::Number(1.0)]),
                Json::String("é/".to_owned())
            ])),
            "nested arrays and escapes"
        );
        assert_eq!(Object::parse("{}").unwrap(), Object::default(), "empty");
    }

    #[test]
    fn malformed_lines_are_refused_where_they_go_wrong() {
        for (line, want) in [
            ("", JsonError::End),
            ("{\"a\":1", JsonError::End),
            ("{\"a\":1} x", JsonError::Unexpected { found: 'x', at: 8 }),
            ("{\"a\":{}}", JsonError::Nested { at: 5 }),
            ("{\"a\":tru}", JsonError::Unexpected { found: 't', at: 5 }),
            ("{\"a\":\"\\q\"}", JsonError::Escape { at: 6 }),
            ("{\"a\":\"\\u12\"}", JsonError::Escape { at: 6 }),
            ("{\"a\":\"\\uD800\"}", JsonError::Escape { at: 6 }),
            (
                "{\"a\":\"\n\"}",
                JsonError::Unexpected { found: '\n', at: 6 },
            ),
            ("{\"a\" 1}", JsonError::Unexpected { found: '1', at: 5 }),
            ("[1]", JsonError::Unexpected { found: '[', at: 0 }),
        ] {
            assert_eq!(Object::parse(line), Err(want), "{line:?}");
        }
    }

    #[test]
    fn numbers_follow_the_json_grammar() {
        for good in [
            "0", "-0", "7", "-12", "1.5", "0.25", "1e3", "1E-3", "-2.5e+10",
        ] {
            assert!(follows_grammar(good), "{good}");
            let line = format!("{{\"x\":{good}}}");
            let parsed = Object::parse(&line).unwrap();
            assert_eq!(
                parsed.get("x").and_then(Json::as_number),
                Some(good.parse().unwrap()),
                "{good}"
            );
        }
        for bad in [
            "", "-", "01", "-01", "1.", ".5", "+1", "1e", "1e+", "1.2.3", "1-2", "--1",
        ] {
            assert!(!follows_grammar(bad), "{bad:?}");
        }
        assert!(
            matches!(
                Object::parse("{\"x\":1e999}"),
                Err(JsonError::Number { at: 5, .. })
            ),
            "past f64's range"
        );
    }
}
