use crate::{
    field::FieldElement,
    list_utils::{remove_trailing_elements, scalar_operation, two_lists_tuple_operation},
};
use itertools::enumerate;

/// Removes zeros from the end of a list.
fn trim_trailing_zeros(p: &[FieldElement]) -> Vec<FieldElement> {
    remove_trailing_elements(p, &FieldElement::zero())
}

pub fn X() -> Polynomial {
    Polynomial::x()
}

/// Represents a polynomial over FieldElement.
#[derive(Debug, PartialEq, Clone)]
pub struct Polynomial(Vec<FieldElement>);

impl Polynomial {
    /// Creates a new Polynomial with the given coefficients.
    /// Internally storing the coefficients in self.poly, least-significant (i.e. free term)
    /// first, so 9 - 3x^2 + 19x^5 is represented internally by the vector [9, 0, -3, 0, 0, 19].
    pub fn new(coefficients: &[FieldElement]) -> Self {
        Polynomial(coefficients.into())
    }

    /// Returns the polynomial x.
    pub fn x() -> Self {
        Polynomial(vec![FieldElement::zero(), FieldElement::one()])
    }

    /// Constructs the monomial coefficient * x^degree.
    pub fn monomial(degree: usize, coefficient: FieldElement) -> Self {
        let mut coefficients = [FieldElement::zero()].repeat(degree);
        coefficients.push(coefficient);
        Polynomial::new(&coefficients)
    }

    /// Generates the polynomial (x-p) for a given point p.
    pub fn gen_linear_term(point: FieldElement) -> Self {
        Polynomial::new(&[FieldElement::zero() - point, FieldElement::one()])
    }

    pub fn modulo(&self, other: Polynomial) -> Polynomial {
        self.qdiv(other).1
    }

    /// The polynomials are represented by a list so the degree is the length of the list minus the
    /// number of trailing zeros (if they exist) minus 1.
    /// This implies that the degree of the zero polynomial will be -1.
    pub fn degree(&self) -> i128 {
        trim_trailing_zeros(&self.0).len() as i128 - 1
    }

    /// Returns the coefficient of x^n
    pub fn get_nth_degree_coefficient(&self, n: usize) -> FieldElement {
        if n as i128 > self.degree() {
            FieldElement::zero()
        } else {
            self.0[n]
        }
    }

    /// Multiplies polynomial by a scalar.
    pub fn scalar_mul(&self, scalar: usize) -> Self {
        Polynomial(scalar_operation(&self.0, |x, y| x * y, scalar))
    }

    /// Evaluates the polynomial at the given point using Horner evaluation.
    pub fn eval(&self, point: FieldElement) -> FieldElement {
        let point: usize = point.val();

        let mut res = 0;

        for coef in self.0.iter().rev() {
            res = (res * point + coef.val()) % FieldElement::k_modulus();
        }

        FieldElement::new(res)
    }

    /// Calculates self^other using repeated squaring.
    pub fn pow(&self, other: usize) -> Self {
        let mut other = other;
        let mut res = Polynomial(vec![FieldElement::one()]);
        let mut current = self.to_owned();
        loop {
            if other % 2 != 0 {
                res = res * current.to_owned();
            }
            other >>= 1;
            if other == 0 {
                break;
            }
            current = current.to_owned() * current;
        }
        res
    }

    /// Composes this polynomial with `other`.
    /// Example:
    /// >>> f = X**2 + X
    /// >>> g = X + 1
    /// >>> f.compose(g) == (2 + 3*X + X**2)
    /// True
    pub fn compose(&self, other: Polynomial) -> Polynomial {
        let mut res = Polynomial::new(&[]);

        for coef in self.0.iter().rev() {
            res = (res * other.clone()) + Polynomial::new(&[*coef]);
        }
        res
    }

    /// Returns q, r the quotient and remainder polynomials respectively, such that
    /// f = q * g + r, where deg(r) < deg(g).
    /// * Assert that g is not the zero polynomial.
    pub fn qdiv(&self, other: impl Into<Polynomial>) -> (Polynomial, Polynomial) {
        let g: Polynomial = other.into();
        let g_coeffs = trim_trailing_zeros(&g.0);
        assert!(!g_coeffs.is_empty(), "Dividing by zero polynomial.");

        let p_coeffs = trim_trailing_zeros(&self.0);
        if p_coeffs.is_empty() {
            return (Polynomial(vec![]), Polynomial(vec![]));
        }

        let mut r = p_coeffs;

        let mut deg_dif = r.len() as i32 - g_coeffs.len() as i32;

        let mut q = if deg_dif.is_negative() {
            vec![FieldElement::zero()]
        } else {
            vec![FieldElement::zero()]
                .repeat(deg_dif as usize + 1)
                .to_vec()
        };

        let q_msc_inv = g_coeffs.last().unwrap().inverse();

        while deg_dif >= 0 {
            let tmp = r.last().unwrap().to_owned() * q_msc_inv;
            q[deg_dif as usize] += tmp;
            let mut last_non_zero = deg_dif - 1;
            for (i, coef) in enumerate(g_coeffs.clone()) {
                let i = i + deg_dif as usize;
                r[i] = r[i] - (tmp * coef);
                if r[i] != FieldElement::zero() {
                    last_non_zero = i as i32;
                }
            }
            // Eliminate trailing zeroes (i.e. make r end with its last non-zero coefficient).
            r = r.into_iter().take((last_non_zero + 1) as usize).collect();
            deg_dif = r.len() as i32 - g_coeffs.len() as i32;
        }

        (Polynomial(trim_trailing_zeros(&q)), Polynomial(r))
    }

    pub fn poly(&self) -> Vec<FieldElement> {
        self.0.clone()
    }
}

impl PartialEq<usize> for Polynomial {
    fn eq(&self, other: &usize) -> bool {
        let fe: FieldElement = (*other).into();
        let poly: Polynomial = fe.into();
        self == &poly
    }
}

impl PartialEq<FieldElement> for Polynomial {
    fn eq(&self, other: &FieldElement) -> bool {
        let other_poly: Polynomial = (*other).into();
        self == &other_poly
    }
}

impl From<usize> for Polynomial {
    fn from(value: usize) -> Self {
        let fe: FieldElement = value.into();
        fe.into()
    }
}

impl From<FieldElement> for Polynomial {
    fn from(value: FieldElement) -> Self {
        Polynomial::new(&[value])
    }
}

impl std::ops::Add for Polynomial {
    type Output = Polynomial;

    fn add(self, other: Self) -> Self::Output {
        Polynomial(two_lists_tuple_operation(
            &self.0,
            &other.0,
            |x, y| x + y,
            FieldElement::zero(),
        ))
    }
}

impl std::ops::AddAssign for Polynomial {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = two_lists_tuple_operation(&self.0, &rhs.0, |x, y| x + y, FieldElement::zero());
    }
}

impl std::ops::Add<usize> for Polynomial {
    type Output = Polynomial;

    fn add(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Add<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn add(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self + other_poly
    }
}

impl std::ops::Sub for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: Self) -> Self::Output {
        Polynomial(two_lists_tuple_operation(
            &self.0,
            &other.0,
            |x, y| x - y,
            FieldElement::zero(),
        ))
    }
}

impl std::ops::Sub<usize> for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self - other_poly
    }
}

impl std::ops::Sub<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn sub(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self - other_poly
    }
}

impl std::ops::Neg for Polynomial {
    type Output = Polynomial;

    fn neg(self) -> Self::Output {
        Polynomial(vec![]) - self
    }
}

impl std::ops::Mul for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: Self) -> Self::Output {
        let mut res = [FieldElement::zero()].repeat((self.degree() + other.degree() + 1) as usize);
        for (i, c1) in self.0.into_iter().enumerate() {
            for (j, c2) in other.clone().0.into_iter().enumerate() {
                res[i + j] += c1 * c2;
            }
        }
        Polynomial(res)
    }
}

impl std::ops::Mul<usize> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self * other_poly
    }
}

impl std::ops::Mul<i128> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: i128) -> Self::Output {
        let other_fe: FieldElement = other.into();
        let other_poly: Polynomial = other_fe.into();
        self * other_poly
    }
}

impl std::ops::Mul<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn mul(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self * other_poly
    }
}

impl std::ops::Div for Polynomial {
    type Output = Polynomial;

    fn div(self, other: Self) -> Self::Output {
        let (div, rem) = self.qdiv(other);
        assert!(
            rem == Polynomial::new(&[]),
            "Polynomials are not divisible."
        );
        div
    }
}

impl std::ops::Div<usize> for Polynomial {
    type Output = Polynomial;

    fn div(self, other: usize) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self / other_poly
    }
}

impl std::ops::Div<FieldElement> for Polynomial {
    type Output = Polynomial;

    fn div(self, other: FieldElement) -> Self::Output {
        let other_poly: Polynomial = other.into();
        self / other_poly
    }
}

/// Given the x_values for evaluating some polynomials, it computes part of the lagrange polynomials
/// required to interpolate a polynomial over this domain.
pub fn calculate_lagrange_polynomials(x_values: &[FieldElement]) -> Vec<Polynomial> {
    let mut lagrange_polynomials: Vec<Polynomial> = vec![];
    let monomials: Vec<Polynomial> = x_values
        .iter()
        .map(|x| Polynomial::monomial(1, FieldElement::one()) - Polynomial::monomial(0, *x))
        .collect();
    let numerator = prod(&monomials);

    for j in 0..x_values.len() {
        // In the denominator, we have:
        // (x_j-x_0)(x_j-x_1)...(x_j-x_{j-1})(x_j-x_{j+1})...(x_j-x_{len(X)-1})
        let mut denoms: Vec<FieldElement> = vec![];
        for (i, x) in x_values.iter().enumerate() {
            if i != j {
                denoms.push(x_values[j] - *x);
            }
        }
        let denominator = prod_field(&denoms);

        // TODO: How to implement the "prod" so that it can handle both "Polynomial" & "Fieldelement".

        // Numerator is a bit more complicated, since we need to compute a poly multiplication here.
        // Similarly to the denominator, we have:
        // (x-x_0)(x-x_1)...(x-x_{j-1})(x-x_{j+1})...(x-x_{len(X)-1})
        let (cur_poly, _) = numerator.qdiv(monomials[j].scalar_mul(denominator.val()));

        lagrange_polynomials.push(cur_poly);
    }

    lagrange_polynomials
}

///    :param y_values: y coordinates of the points.
///    :param lagrange_polynomials: the polynomials obtained from calculate_lagrange_polynomials.
///    :return: the interpolated poly/
pub fn interpolate_poly_lagrange(
    y_values: &[FieldElement],
    lagrange_polynomials: Vec<Polynomial>,
) -> Polynomial {
    let mut poly = Polynomial::new(&[]);

    for (j, y_value) in y_values.iter().enumerate() {
        poly += lagrange_polynomials[j].scalar_mul(y_value.val());
    }

    poly
}
///    Returns a polynomial of degree < len(x_values) that evaluates to y_values[i] on x_values[i] for
///    all i.
pub fn interpolate_poly(x_values: &[FieldElement], y_values: &[FieldElement]) -> Polynomial {
    assert!(x_values.len() == y_values.len());

    let lp = calculate_lagrange_polynomials(x_values);

    interpolate_poly_lagrange(y_values, lp)
}

/// Computes a product
pub fn prod(values: &[Polynomial]) -> Polynomial {
    let values_len = values.len();

    if values_len == 0 {
        return Polynomial::new(&[]);
    }

    if values_len == 1 {
        return values[0].clone();
    }

    prod(&values[0..values.len() / 2]) * prod(&values[values.len() / 2..])
}

/// Computes a product of [FieldElement]
pub fn prod_field(values: &[FieldElement]) -> FieldElement {
    let values_len = values.len();

    if values_len == 0 {
        return FieldElement::one();
    }

    if values_len == 1 {
        return values[0];
    }

    prod_field(&values[0..values.len() / 2]) * prod_field(&values[values.len() / 2..])
}

#[cfg(test)]
mod tests {
    use super::{interpolate_poly, prod, Polynomial, X};
    use crate::field::FieldElement;
    use itertools::Itertools;

    /// Returns a random polynomial of a prescribed degree which is not the zero polynomial.
    fn generate_random_polynomail(degree: usize) -> Polynomial {
        let leading = FieldElement::random_element(&[FieldElement::zero()]);
        let mut elems = (1..degree)
            .into_iter()
            .map(|_| FieldElement::random_element(&[]))
            .collect_vec();
        elems.push(leading);
        Polynomial(elems)
    }

    #[test]
    fn test_poly_mul() {
        let result = (X() + 1) * (X() + 1);
        let expected = X().pow(2) + X() * 2usize + 1;
        assert_eq!(result, expected)
    }

    #[test]
    fn test_div() {
        let p = X().pow(2) - 1;
        assert_eq!(p / (X() - 1), X() + 1)
    }

    #[test]
    fn test_modulo() {
        let p: Polynomial = X().pow(9) - X() * 5usize + 4;
        assert_eq!(p.modulo(X().pow(2) + 1), X() * (-4i128) + 4)
    }

    #[test]
    fn test_qdiv() {
        let p: Polynomial = X().pow(2) - X() * 2usize + 1;
        let g: Polynomial = X() - 1;
        let (q, r) = p.qdiv(g.clone());
        assert!(g == q);
        assert!(r == Polynomial(vec![]));
    }

    #[test]
    fn test_interpolate_poly() {
        let x_values = vec![
            FieldElement::new(1),
            FieldElement::new(3),
            FieldElement::from(-2_i128),
        ];
        let y_values = vec![
            FieldElement::new(12),
            FieldElement::new(10),
            FieldElement::from(-15_i128),
        ];

        let poly = interpolate_poly(&x_values, &y_values);

        assert!(
            poly == Polynomial::new(&[
                FieldElement::from(7_i128),
                FieldElement::from(7_i128),
                FieldElement::from(-2_i128)
            ])
        );
    }

    #[test]
    fn test_prod_polys() {
        let polys = vec![
            Polynomial::new(&[FieldElement::new(1), FieldElement::new(1)]),
            Polynomial::new(&[FieldElement::new(1), FieldElement::new(1)]),
            Polynomial::new(&[FieldElement::new(1), FieldElement::new(1)]),
        ];

        let res = prod(&polys);

        assert!(
            res == Polynomial::new(&[
                FieldElement::new(1),
                FieldElement::new(3),
                FieldElement::new(3),
                FieldElement::new(1),
            ])
        );
    }
}
