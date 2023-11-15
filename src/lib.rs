use gen_math_lib::{
    integration::{integrate, rk7_step},
    progression::arithmetic_bounded,
};
use std::ops::{Add, Mul};

pub fn cornulike<'inp>(
    from: f64,
    to: f64,
    a0: f64,
    curvature: impl Fn(f64) -> f64 + 'inp,
) -> impl Iterator<Item = (f64, (f64, f64))> + 'inp {
    #[derive(Debug, Clone)]
    struct S(f64, (f64, f64));
    impl Add for S {
        type Output = S;

        fn add(self, rhs: Self) -> Self::Output {
            S(self.0 + rhs.0, (self.1 .0 + rhs.1 .0, self.1 .1 + rhs.1 .1))
        }
    }
    impl Mul<f64> for S {
        type Output = S;

        fn mul(self, rhs: f64) -> Self::Output {
            S(self.0 * rhs, (self.1 .0 * rhs, self.1 .1 * rhs))
        }
    }
    let dt = 0.000001;
    integrate(
        S(a0, (0.0, 0.0)),
        from,
        move |S(a, _), s| S(curvature(s), (a.cos(), a.sin())),
        arithmetic_bounded(from, to, dt).skip(1),
        rk7_step,
    )
    .step_by(1000)
    .map(|(S(_, (x, y)), s)| (s, (x, y)))
}

fn weierstrass(x: f64) -> f64 {
    (0..20)
        .map(|n| 2.0f64.powi(n) * (2.0f64.powi(n) * std::f64::consts::PI * x).cos())
        .sum()
}

#[cfg(test)]
mod tests {
    use std::{
        fs::OpenOptions,
        io::{BufWriter, Write},
        u128::MAX,
    };

    use gen_math_lib::{memoized, progression::arithmetic_bounded};

    use crate::{cornulike, weierstrass};

    #[test]
    fn cornu() {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open("output.txt")
            .expect("Should open a file");
        for (s, (x, y)) in cornulike(0.0, 10.0, 0.0, weierstrass) {
            file.write_fmt(format_args!("{s} {x} {y}\n"))
                .expect("Should be able to write");
        }
    }

    #[test]
    fn fun() {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open("fun.txt")
            .expect("Should open a file");
        for x in arithmetic_bounded(0.0, 1.0, 0.001) {
            let y: f64 = weierstrass(x);
            file.write_fmt(format_args!("{x} {y}\n"))
                .expect("Should be able to write");
        }
    }
}
