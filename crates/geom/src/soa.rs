//! [`Vec3`] arithmetic over whole rows of vectors, laid out as structure of arrays.
//!
//! Frames and normals run a short formula (a cross product, a normalization, a projection) on
//! every vertex of a row. Done vertex by vertex, each step waits for the one before it (a square
//! root, then a division, then the next product), and the core idles. Done a step at a time
//! over the whole row, with each component in its own array, every step is one loop over
//! independent vertices that the compiler vectorises and pipelines.
//!
//! Every operation does, on each vertex, exactly the `f32` operations `Vec3`'s does, in the same
//! order and grouping, so vertex `k` of a result has the bits `Vec3` gives for vertex `k`'s
//! inputs. Nothing is fused or reordered: Rust never contracts `a * b + c` into a fused
//! multiply-add, and these loops never call `mul_add`. The generated code of the hot loops was
//! checked for FMA instructions when this was written (M0c2 Task 4b).

use crate::Vec3;

/// A row of vectors, one array per component.
#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Vectors {
    /// The x components.
    x: Vec<f32>,
    /// The y components.
    y: Vec<f32>,
    /// The z components.
    z: Vec<f32>,
}

impl Vectors {
    /// Makes the row `points`, in order.
    pub(crate) fn set_points(&mut self, points: &[Vec3]) {
        self.x.clear();
        self.x.extend(points.iter().map(|p| p.x));
        self.y.clear();
        self.y.extend(points.iter().map(|p| p.y));
        self.z.clear();
        self.z.extend(points.iter().map(|p| p.z));
    }

    /// Makes the row `plus[k] − minus[k]` for every `k` (as many as the shorter slice holds):
    /// [`Vec3`]'s subtraction, component by component.
    pub(crate) fn set_difference(&mut self, plus: &[Vec3], minus: &[Vec3]) {
        let pairs = || plus.iter().zip(minus);
        self.x.clear();
        self.x.extend(pairs().map(|(p, m)| p.x - m.x));
        self.y.clear();
        self.y.extend(pairs().map(|(p, m)| p.y - m.y));
        self.z.clear();
        self.z.extend(pairs().map(|(p, m)| p.z - m.z));
    }

    /// Makes the row `a[k] × b[k]`: [`Vec3::cross`], vector by vector.
    pub(crate) fn set_cross(&mut self, a: &Self, b: &Self) {
        // One component: `p·q − r·s`.
        let component = |out: &mut Vec<f32>, (p, q): (&[f32], &[f32]), (r, s): (&[f32], &[f32])| {
            out.clear();
            let terms = p.iter().zip(q).zip(r.iter().zip(s));
            out.extend(terms.map(|((&pk, &qk), (&rk, &sk))| pk * qk - rk * sk));
        };
        component(&mut self.x, (&a.y, &b.z), (&a.z, &b.y));
        component(&mut self.y, (&a.z, &b.x), (&a.x, &b.z));
        component(&mut self.z, (&a.x, &b.y), (&a.y, &b.x));
    }

    /// `a[k] · b[k]` into `out`: [`Vec3::dot`], `x·x' + y·y' + z·z'` summed left to right.
    pub(crate) fn dot_into(a: &Self, b: &Self, out: &mut Vec<f32>) {
        out.clear();
        let xs = a.x.iter().zip(&b.x);
        let ys = a.y.iter().zip(&b.y);
        let zs = a.z.iter().zip(&b.z);
        out.extend(
            xs.zip(ys)
                .zip(zs)
                .map(|(((&x, &ox), (&y, &oy)), (&z, &oz))| x * ox + y * oy + z * oz),
        );
    }

    /// Normalizes every vector in place with [`Vec3::normalized`]'s arithmetic (scaled by the
    /// reciprocal of its length) and writes the lengths to `lengths`.
    ///
    /// A vector's result is what `Vec3::normalized` returns where
    /// [`crate::math::normalizable`] accepts its length; elsewhere it is whatever the arithmetic
    /// gave, for the caller to discard.
    pub(crate) fn normalize(&mut self, lengths: &mut Vec<f32>) {
        Self::dot_into(self, self, lengths);
        for length in lengths.iter_mut() {
            *length = length.sqrt();
        }
        let components = self.x.iter_mut().zip(&mut self.y).zip(&mut self.z);
        for (((x, y), z), &length) in components.zip(lengths.iter()) {
            let r = length.recip();
            (*x, *y, *z) = (*x * r, *y * r, *z * r);
        }
    }

    /// Makes the row `a[k] − n[k] · s[k]`: [`Vec3`]'s `a - n * s`, vector by vector.
    pub(crate) fn set_minus_scaled(&mut self, a: &Self, n: &Self, s: &[f32]) {
        let minus_scaled = |out: &mut Vec<f32>, ac: &[f32], nc: &[f32]| {
            out.clear();
            out.extend(
                ac.iter()
                    .zip(nc)
                    .zip(s)
                    .map(|((&ak, &nk), &sk)| ak - nk * sk),
            );
        };
        minus_scaled(&mut self.x, &a.x, &n.x);
        minus_scaled(&mut self.y, &a.y, &n.y);
        minus_scaled(&mut self.z, &a.z, &n.z);
    }

    /// The vectors, in order.
    pub(crate) fn points(&self) -> impl Iterator<Item = Vec3> {
        self.x
            .iter()
            .zip(&self.y)
            .zip(&self.z)
            .map(|((&x, &y), &z)| Vec3::new(x, y, z))
    }
}

#[cfg(test)]
#[expect(clippy::missing_panics_doc, reason = "tests panic to fail")]
mod tests {
    use super::*;
    use crate::math::normalizable;
    use crate::test_rng::Rng;

    /// Values that exercise every edge of the arithmetic: random, tiny, huge, zero, infinite, NaN.
    fn awkward(rng: &mut Rng, k: usize) -> Vec3 {
        match k % 8 {
            0 => Vec3::ZERO,
            1 => Vec3::new(1e-30, -1e-25, 1e-38),
            2 => Vec3::new(3e18, -2e19, 1e19),
            3 => Vec3::new(f32::INFINITY, 0.0, 1.0),
            4 => Vec3::new(0.5, f32::NAN, 0.0),
            _ => rng.vec3(1.0),
        }
    }

    /// The bits of a row of points.
    fn bits(points: impl Iterator<Item = Vec3>) -> Vec<[u32; 3]> {
        points.map(Vec3::to_bits).collect()
    }

    #[test]
    #[expect(
        clippy::many_single_char_names,
        reason = "operands keep the names of the operations they test"
    )]
    fn every_vector_has_the_bits_of_vec3() {
        let mut rng = Rng::new(41);
        let a: Vec<Vec3> = (0..1000).map(|k| awkward(&mut rng, k)).collect();
        let b: Vec<Vec3> = (0..1000).map(|k| awkward(&mut rng, 3 * k + 1)).collect();
        let c: Vec<Vec3> = (0..1000).map(|k| awkward(&mut rng, 5 * k + 2)).collect();
        let row = |plus: &[Vec3]| {
            let mut v = Vectors::default();
            v.set_difference(plus, &vec![Vec3::ZERO; plus.len()]);
            v
        };
        let (va, vb, vc) = (row(&a), row(&b), row(&c));
        assert_eq!(bits(va.points()), bits(a.iter().copied()), "a − 0 is a");
        let mut loaded = Vectors::default();
        loaded.set_points(&a);
        assert_eq!(bits(loaded.points()), bits(a.iter().copied()), "set_points");

        let mut diff = Vectors::default();
        diff.set_difference(&a, &b);
        let differences = a.iter().zip(&b).map(|(&p, &q)| p - q);
        assert_eq!(bits(diff.points()), bits(differences), "difference");

        let mut cross = Vectors::default();
        cross.set_cross(&va, &vb);
        let crosses = a.iter().zip(&b).map(|(&p, &q)| p.cross(q));
        assert_eq!(bits(cross.points()), bits(crosses), "cross");

        let mut dot = Vec::new();
        Vectors::dot_into(&va, &vb, &mut dot);
        let dots: Vec<u32> = a
            .iter()
            .zip(&b)
            .map(|(&p, &q)| p.dot(q).to_bits())
            .collect();
        assert_eq!(
            dot.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            dots,
            "dot"
        );

        let s: Vec<f32> = c.iter().map(|p| p.x).collect();
        let mut minus = Vectors::default();
        minus.set_minus_scaled(&va, &vb, &s);
        let scaled = a.iter().zip(&b).zip(&s).map(|((&p, &q), &sk)| p - q * sk);
        assert_eq!(bits(minus.points()), bits(scaled), "minus scaled");

        let mut unit = vc;
        let mut lengths = Vec::new();
        unit.normalize(&mut lengths);
        for (k, ((p, &length), &original)) in unit.points().zip(&lengths).zip(&c).enumerate() {
            match original.normalized() {
                Some(n) => {
                    assert!(normalizable(length), "{k}: normalizes");
                    assert_eq!(p.to_bits(), n.to_bits(), "{k}: normalized");
                }
                None => assert!(!normalizable(length), "{k}: doesn't normalize"),
            }
        }
        assert_eq!(lengths.len(), c.len(), "one length per vector");
    }
}
