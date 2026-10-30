//! The MatCap images, generated: a few studio-lit spheres, ours.
//!
//! Each [`MatCapStyle`] is a sphere seen head on, in the view's frame (x right, y up, z towards
//! the eye), lit by the same small studio, linear, HDR (a glossy highlight runs past 1):
//!
//! - a **key** light above and to the left, a little in front, warm;
//! - a **fill** from the right, cool and dim;
//! - a **rim** from behind and above, which a wrapped diffuse term brings round to the edge;
//! - a **sky** over the ground: a hemisphere that is lighter above than below.
//!
//! A texel `(s, t)` of a `side × side` image holds the sphere where the normal is `(x, y,
//! √(1 − x² − y²))`, `x = 2 (s + ½) / side − 1`, `y = 1 − 2 (t + ½) / side`; past the disc's edge
//! the normal is the edge's (`z = 0`), so filtering near the edge never reads anything else. The
//! mip levels are each the one before averaged over 2 × 2 texels, in linear colour.

/// The images' side, texels.
pub(super) const SIDE: u32 = 256;

/// Their mip levels: down to 1 × 1.
pub(super) const LEVELS: u32 = SIDE.ilog2() + 1;

/// A MatCap style: one of the generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MatCapStyle {
    /// Neutral clay: a light, warm grey with a broad, soft sheen.
    #[default]
    Clay,
    /// Glossy: a dark blue-grey with sharp highlights and a bright reflection at grazing angles.
    Glossy,
    /// Pale: near white and evenly lit, for reading fine detail.
    Pale,
}

impl MatCapStyle {
    /// Every style, in layer order.
    pub const ALL: [Self; 3] = [Self::Clay, Self::Glossy, Self::Pale];

    /// The style's layer in the images.
    #[must_use]
    pub(crate) const fn layer(self) -> u32 {
        match self {
            Self::Clay => 0,
            Self::Glossy => 1,
            Self::Pale => 2,
        }
    }

    /// The next style, round to the first.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::Clay => Self::Glossy,
            Self::Glossy => Self::Pale,
            Self::Pale => Self::Clay,
        }
    }

    /// Its name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clay => "clay",
            Self::Glossy => "glossy",
            Self::Pale => "pale",
        }
    }

    /// How the style reflects light.
    const fn finish(self) -> Finish {
        match self {
            Self::Clay => Finish {
                albedo: [0.60, 0.54, 0.48],
                wrap: 0.12,
                ambient: 0.8,
                gloss: 12.0,
                sheen: 0.08,
                mirror: 0.0,
                edge: 0.45,
            },
            Self::Glossy => Finish {
                albedo: [0.10, 0.115, 0.14],
                wrap: 0.1,
                ambient: 0.6,
                gloss: 160.0,
                sheen: 1.0,
                mirror: 0.6,
                edge: 0.0,
            },
            Self::Pale => Finish {
                albedo: [0.80, 0.80, 0.78],
                wrap: 0.5,
                ambient: 1.4,
                gloss: 8.0,
                sheen: 0.02,
                mirror: 0.0,
                edge: 0.25,
            },
        }
    }

    /// The sphere's linear colour where its unit normal is `n` (view space, z towards the eye).
    fn radiance(self, n: [f32; 3]) -> [f32; 3] {
        let finish = self.finish();
        let towards_eye = [0.0, 0.0, 1.0];
        let lights = [KEY, FILL, RIM];
        let mut colour = sky(n[1]).map(|c| c * finish.ambient);
        for light in lights {
            let facing = dot(n, light.direction);
            let diffuse = ((facing + finish.wrap) / (1.0 + finish.wrap)).max(0.0);
            let half = normalized(add(light.direction, towards_eye));
            let gloss = finish.gloss;
            let highlight = if facing > 0.0 {
                dot(n, half).max(0.0).powf(gloss) * (gloss + 8.0) / (8.0 * std::f32::consts::PI)
            } else {
                0.0
            };
            let channels = colour
                .iter_mut()
                .zip(finish.albedo.iter().zip(light.colour));
            for (c, (albedo, tint)) in channels {
                *c += tint * light.power * (albedo * diffuse + finish.sheen * highlight);
            }
        }
        // A studio's reflection at grazing angles (Schlick's Fresnel term over the sky).
        let facing_eye = n[2].clamp(0.0, 1.0);
        let fresnel = 0.04 + 0.96 * (1.0 - facing_eye).powi(5);
        let reflected_up = 2.0 * facing_eye * n[1];
        let reflection = sky(reflected_up).map(|c| c * 4.0);
        let mirror = finish.mirror * fresnel;
        let reflected = colour.iter_mut().zip(reflection);
        for (c, r) in reflected {
            *c += mirror * r;
        }
        // Clay darkens towards the outline, as a matte surface does where it turns away.
        let turned = 1.0 - facing_eye;
        colour.map(|c| c * (1.0 - finish.edge * turned * turned))
    }
}

/// How a style reflects light.
#[derive(Debug, Clone, Copy)]
struct Finish {
    /// Diffuse colour, linear.
    albedo: [f32; 3],
    /// How far the diffuse term wraps past the terminator: 0 is Lambert's.
    wrap: f32,
    /// The sky's strength.
    ambient: f32,
    /// The highlight's sharpness (a Blinn-Phong exponent).
    gloss: f32,
    /// The highlight's strength.
    sheen: f32,
    /// The grazing reflection's strength.
    mirror: f32,
    /// How much darker the outline is than the middle.
    edge: f32,
}

/// A light of the studio.
#[derive(Debug, Clone, Copy)]
struct Light {
    /// Unit length, towards the light, view space.
    direction: [f32; 3],
    /// Its colour, linear.
    colour: [f32; 3],
    /// Its strength.
    power: f32,
}

/// The key light: above left, a little in front, warm.
const KEY: Light = Light {
    direction: [-0.474, 0.685, 0.553],
    colour: [1.0, 0.95, 0.88],
    power: 1.0,
};

/// The fill: from the right, cool, dim.
const FILL: Light = Light {
    direction: [0.752, 0.0, 0.659],
    colour: [0.80, 0.88, 1.0],
    power: 0.3,
};

/// The rim: from behind and above.
const RIM: Light = Light {
    direction: [0.124, 0.496, -0.859],
    colour: [1.0, 1.0, 1.0],
    power: 0.45,
};

/// The sky's colour, linear, seen along a direction whose up component is `up`: the ground's
/// below, the sky's above.
fn sky(up: f32) -> [f32; 3] {
    let ground = [0.030, 0.028, 0.026];
    let above = [0.11, 0.115, 0.125];
    let t = (up * 0.5 + 0.5).clamp(0.0, 1.0);
    let mut colour = ground;
    let towards = colour.iter_mut().zip(above);
    for (c, high) in towards {
        *c += (high - *c) * t;
    }
    colour
}

/// `a · b`.
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `a + b`.
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// `v` scaled to unit length (`v` itself if it is zero).
fn normalized(v: [f32; 3]) -> [f32; 3] {
    let length = dot(v, v).sqrt();
    if length > 0.0 {
        v.map(|x| x / length)
    } else {
        v
    }
}

/// One mip level of a MatCap image: linear RGBA, row by row from the top.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Image {
    /// Its side, texels.
    side: u32,
    /// Its texels.
    texels: Vec<[f32; 4]>,
}

impl Image {
    /// Its side, texels.
    pub(super) const fn side(&self) -> u32 {
        self.side
    }

    /// Its texels.
    #[cfg(test)]
    pub(super) fn texels(&self) -> &[[f32; 4]] {
        &self.texels
    }

    /// Its texels as `Rgba16Float` holds them.
    pub(super) fn half_texels(&self) -> Vec<[u16; 4]> {
        self.texels
            .iter()
            .map(|texel| texel.map(half_bits))
            .collect()
    }

    /// The level below: each texel the mean of the 2 × 2 above it.
    fn halved(&self) -> Self {
        let side = (self.side / 2).max(1);
        let from = usize::try_from(self.side).unwrap_or(usize::MAX);
        let to = usize::try_from(side).unwrap_or(0);
        let texel = |s: usize, t: usize| {
            let at = t.min(from - 1) * from + s.min(from - 1);
            self.texels.get(at).copied().unwrap_or_default()
        };
        let grid = (0..to).flat_map(|t| (0..to).map(move |s| (s, t)));
        let means = grid.map(|(s, t)| {
            let corners =
                [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(ds, dt)| texel(2 * s + ds, 2 * t + dt));
            let mut mean = [0.0; 4];
            for corner in corners {
                let channels = mean.iter_mut().zip(corner);
                for (m, c) in channels {
                    *m += c / 4.0;
                }
            }
            mean
        });
        Self {
            side,
            texels: means.collect(),
        }
    }
}

/// The sphere's unit normal at `(across, up)` of the image, both in `-1..1`; past the disc's edge,
/// the edge's.
fn sphere_normal(across: f32, up: f32) -> [f32; 3] {
    let reach = across * across + up * up;
    if reach >= 1.0 {
        let radius = reach.sqrt();
        [across / radius, up / radius, 0.0]
    } else {
        [across, up, (1.0 - reach).sqrt()]
    }
}

/// `style`'s image and every mip level of it, largest first.
pub(super) fn mips(style: MatCapStyle) -> Vec<Image> {
    let side = SIDE;
    let scale = 2.0 / f32::from(u16::try_from(side).unwrap_or(u16::MAX));
    let centre = |k: u32| f32::from(u16::try_from(k).unwrap_or(u16::MAX)) + 0.5;
    let grid = (0..side).flat_map(|row| (0..side).map(move |column| (column, row)));
    let texels = grid.map(|(column, row)| {
        let normal = sphere_normal(centre(column) * scale - 1.0, 1.0 - centre(row) * scale);
        let [red, green, blue] = style.radiance(normal);
        [red, green, blue, 1.0]
    });
    let mut levels = vec![Image {
        side,
        texels: texels.collect(),
    }];
    while let Some(last) = levels.last()
        && last.side > 1
    {
        let next = last.halved();
        levels.push(next);
    }
    levels
}

/// `value` as an IEEE 754 half (binary16), rounded to the nearest, ties to even: `Rgba16Float`'s
/// texels. Past the largest half it is infinite; NaN stays NaN.
pub(super) fn half_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = u16::try_from((bits >> 16) & 0x8000).unwrap_or(0);
    let exponent = (bits >> 23) & 0xFF;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        let nan = if mantissa == 0 { 0 } else { 0x200 };
        return sign | 0x7C00 | nan;
    }
    // The half's biased exponent: f32's bias is 127, the half's 15.
    let Some(biased) = exponent.checked_sub(112).filter(|&e| e > 0) else {
        // Subnormal or zero as a half: the mantissa, implicit bit included, shifted to units
        // of 2^-24, rounded to even. Below half the smallest subnormal it is zero.
        let shift = 126 - exponent;
        if shift > 24 {
            return sign;
        }
        let m = mantissa | 0x80_0000;
        let rounded = (m + (1 << (shift - 1)) - 1 + ((m >> shift) & 1)) >> shift;
        return sign | u16::try_from(rounded).unwrap_or(0);
    };
    if biased >= 0x1F {
        return sign | 0x7C00;
    }
    // Ten bits of mantissa, rounded to even; a carry runs into the exponent, as it should.
    let rounded = mantissa + 0xFFF + ((mantissa >> 13) & 1);
    let half = (biased << 10) + (rounded >> 13);
    sign | u16::try_from(half.min(0x7C00)).unwrap_or(0x7C00)
}
