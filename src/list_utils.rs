use crate::field::FieldElement;
use itertools::{
    EitherOrBoth::{self},
    Itertools,
};
use std::iter::Iterator;

pub fn remove_trailing_elements(
    elements: &[FieldElement],
    element_to_remove: &FieldElement,
) -> Vec<FieldElement> {
    let filtered_iter = elements
        .iter()
        .rev()
        .skip_while(|x| *x == element_to_remove)
        .map(Clone::clone);
    let mut res = filtered_iter.collect::<Vec<FieldElement>>();
    res.reverse();
    res
}

pub fn two_lists_tuple_operation<F>(
    l1: &[FieldElement],
    l2: &[FieldElement],
    operation: F,
    fill_value: FieldElement,
) -> Vec<FieldElement>
where
    F: Fn(FieldElement, FieldElement) -> FieldElement,
{
    l1.iter()
        .zip_longest(l2)
        .map(|x| match x {
            EitherOrBoth::Both(e1, e2) => operation(*e1, *e2),
            EitherOrBoth::Left(e) => operation(*e, fill_value),
            EitherOrBoth::Right(e) => operation(fill_value, *e),
        })
        .collect()
}

pub fn scalar_operation<F>(
    elements: &[FieldElement],
    operation: F,
    scalar: impl Into<FieldElement>,
) -> Vec<FieldElement>
where
    F: Fn(FieldElement, FieldElement) -> FieldElement,
{
    let value: FieldElement = scalar.into();
    elements.iter().map(|e| operation(*e, value)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_trailing_elements() {
        let f = vec![
            FieldElement::new(3),
            FieldElement::new(4),
            FieldElement::new(0),
            FieldElement::new(1),
        ];
        let g = vec![
            FieldElement::new(3),
            FieldElement::new(4),
            FieldElement::new(0),
            FieldElement::new(1),
            FieldElement::new(0),
            FieldElement::new(0),
            FieldElement::new(0),
        ];

        let res = remove_trailing_elements(&g, &FieldElement::new(0));
        assert_eq!(res, f);
    }

    #[test]
    fn test_two_lists_tuple_operation() {
        let f = vec![
            FieldElement::new(3),
            FieldElement::new(4),
            FieldElement::new(0),
            FieldElement::new(1),
        ];
        let g = vec![
            FieldElement::new(3),
            FieldElement::new(4),
            FieldElement::new(0),
            FieldElement::new(1),
            FieldElement::new(6),
        ];
        let fill_value = FieldElement::zero();

        let res = two_lists_tuple_operation(&f, &g, |x, y| x + y, fill_value);

        assert_eq!(
            res,
            vec![
                FieldElement::new(6),
                FieldElement::new(8),
                FieldElement::new(0),
                FieldElement::new(2),
                FieldElement::new(6),
            ]
        )
    }

    #[test]
    fn test_scalar_operation() {
        let poly_coeffs = vec![
            FieldElement::new(3),
            FieldElement::new(4),
            FieldElement::new(0),
            FieldElement::new(1),
        ];
        let scalar = FieldElement::new(1);

        let res = scalar_operation(&poly_coeffs, |x, y| x * y, scalar);
        assert_eq!(res, poly_coeffs);
    }
}
