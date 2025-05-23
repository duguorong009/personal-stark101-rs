use rand::Rng;

/// An implementation of field elements from F_(3 * 2**30 + 1).
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct FieldElement(usize);

impl FieldElement {
    pub fn new(value: usize) -> Self {
        FieldElement(value % FieldElement::k_modulus())
    }

    pub fn k_modulus() -> usize {
        3 * 2usize.pow(30) + 1
    }

    pub fn generator() -> Self {
        FieldElement(5)
    }

    /// Obtains the zero element of the field.
    pub fn zero() -> Self {
        FieldElement(0)
    }

    /// Obtains the unit element of the field.
    pub fn one() -> Self {
        FieldElement(1)
    }

    pub fn inverse(&self) -> Self {
        let (mut t, mut new_t) = (0_i128, 1_i128);
        let (mut r, mut new_r) = (FieldElement::k_modulus() as i128, self.0 as i128);
        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - (quotient * new_t));
            (r, new_r) = (new_r, r - quotient * new_r);
        }
        assert!(r == 1);
        t.into()
    }

    pub fn pow(&self, n: usize) -> Self {
        let mut n = n;
        let mut current_pow = self.to_owned();
        let mut res = FieldElement::one();
        while n > 0 {
            if n % 2 != 0 {
                res *= current_pow;
            }
            n /= 2;
            current_pow *= current_pow;
        }
        res
    }

    /// Naively checks that the element is of order n by raising it to all powers up to n, checking
    /// that the element to the n-th power is the unit, but not so for any k < n.
    pub fn is_order(&self, n: usize) -> bool {
        assert!(n >= 1);
        let mut h = FieldElement(1);
        for _ in 1..n {
            h *= self;
            if h == FieldElement::one() {
                return false;
            }
        }
        h * self == FieldElement::one()
    }

    pub fn random_element(exclude_elements: &[FieldElement]) -> FieldElement {
        let mut rnd = rand::rng();
        let random_element: usize = rnd.random_range(0..FieldElement::k_modulus());
        let mut candidate = FieldElement::new(random_element);

        while exclude_elements.contains(&candidate) {
            let random_element: usize = rnd.random_range(0..FieldElement::k_modulus());
            candidate = FieldElement::new(random_element);
        }

        candidate
    }

    pub fn val(&self) -> usize {
        self.0
    }
}

impl PartialEq<usize> for FieldElement {
    fn eq(&self, other: &usize) -> bool {
        self == &FieldElement::new(*other)
    }
}

impl std::ops::Add for FieldElement {
    type Output = FieldElement;

    fn add(self, rhs: Self) -> Self::Output {
        FieldElement::new(self.0 + rhs.0)
    }
}

impl std::ops::Add for &FieldElement {
    type Output = FieldElement;

    fn add(self, rhs: Self) -> Self::Output {
        FieldElement::new(self.0 + rhs.0)
    }
}

impl std::ops::AddAssign for FieldElement {
    fn add_assign(&mut self, rhs: Self) {
        *self = FieldElement::new(self.0 + rhs.0)
    }
}

impl std::ops::Mul for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: Self) -> Self::Output {
        FieldElement::new(self.0 * rhs.0)
    }
}

impl std::ops::Mul<&FieldElement> for FieldElement {
    type Output = FieldElement;

    fn mul(self, rhs: &Self) -> Self::Output {
        FieldElement::new(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign for FieldElement {
    fn mul_assign(&mut self, rhs: Self) {
        *self = FieldElement::new(self.0 * rhs.0)
    }
}

impl std::ops::MulAssign<&FieldElement> for FieldElement {
    fn mul_assign(&mut self, rhs: &Self) {
        *self = FieldElement::new(self.0 * rhs.0)
    }
}

impl std::ops::Sub for FieldElement {
    type Output = FieldElement;

    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 as i128 - rhs.0 as i128).into()
    }
}

impl std::ops::Sub<&FieldElement> for FieldElement {
    type Output = FieldElement;

    fn sub(self, rhs: &Self) -> Self::Output {
        (self.0 as i128 - rhs.0 as i128).into()
    }
}

impl std::ops::Div for FieldElement {
    type Output = FieldElement;

    fn div(self, rhs: Self) -> Self::Output {
        self * rhs.inverse()
    }
}

impl std::ops::Div<usize> for FieldElement {
    type Output = FieldElement;

    fn div(self, rhs: usize) -> Self::Output {
        self * FieldElement::new(rhs).inverse()
    }
}

impl std::ops::Neg for FieldElement {
    type Output = FieldElement;

    fn neg(self) -> Self::Output {
        FieldElement::zero() - self
    }
}

impl std::fmt::Display for FieldElement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<i128> for FieldElement {
    fn from(value: i128) -> Self {
        let value_mod_p = if value > 0 {
            value % (FieldElement::k_modulus() as i128)
        } else {
            value + FieldElement::k_modulus() as i128
        };
        FieldElement::new(value_mod_p.try_into().unwrap())
    }
}

impl From<usize> for FieldElement {
    fn from(value: usize) -> Self {
        FieldElement::new(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_operations() {
        let t = FieldElement::new(2).pow(30) * FieldElement::new(3) + FieldElement::new(1);
        assert!(t == FieldElement::zero());
    }

    #[test]
    fn test_field_div() {
        for _ in 0..1000 {
            let t = FieldElement::random_element(&[FieldElement::zero()]);
            let t_inv = FieldElement::one() / t;
            assert!(t_inv == t.inverse());
            assert!(t_inv * t == FieldElement::one());
        }
    }

    #[test]
    fn inverse_test() {
        let x = FieldElement::new(2);
        let x_inv = x.inverse();

        assert_eq!(FieldElement::one(), x * x_inv)
    }
}
