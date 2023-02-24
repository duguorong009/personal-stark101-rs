use crate::field::FieldElement;
use itertools::{EitherOrBoth::*, Itertools};
use std::iter::Iterator;

pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

// pub fn remove_trailing_elements(
//     list_of_elements: &[FieldElement],
//     element_to_remove: FieldElement,
// ) -> Vec<FieldElement> {
//     let res: Vec<FieldElement> = list_of_elements.iter().map(|x| x.clone()).rev().collect();

//     let res = res..skip_while(|x| x == &FieldElement::zero()).rev();
// }

pub fn remove_trailing_elements(
    list_of_elements: &[FieldElement],
    element_to_remove: FieldElement,
) -> Vec<FieldElement> {
    let reversed_list = list_of_elements.iter().rev();
    let filtered_list = reversed_list.skip_while(|&x| x == &element_to_remove);

    // TODO: Need to impl Iter for FieldElement ?
    let mut res = vec![];
    for element in filtered_list {
        res.push(element.clone());
    }
    res.reverse();
    res
}

pub fn two_lists_tuple_operation(
    f: &[FieldElement],
    g: &[FieldElement],
    operation: Operator,
    fill_value: FieldElement,
) -> Vec<FieldElement> {
    let mut res = vec![];

    let op = |l: FieldElement, r: FieldElement| -> FieldElement {
        match operation {
            Operator::Add => l + r,
            Operator::Sub => l - r,
            Operator::Mul => l * r,
            Operator::Div => l / r,
        }
    };

    for pair in f.iter().zip_longest(g.iter()) {
        let temp = match pair {
            Both(l, r) => op(l.clone(), r.clone()),
            Left(l) => op(l.clone(), fill_value),
            Right(r) => op(fill_value, r.clone()),
        };
        res.push(temp);
    }
    res
}

pub fn scalar_operation(
    list_of_elements: &[FieldElement],
    operation: Operator,
    scalar: FieldElement,
) -> Vec<FieldElement> {
    list_of_elements
        .iter()
        .map(|a| match operation {
            Operator::Add => a.clone() + scalar,
            Operator::Sub => a.clone() - scalar,
            Operator::Mul => a.clone() * scalar,
            Operator::Div => a.clone() / scalar,
        })
        .collect()
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

        let res = remove_trailing_elements(&g, FieldElement::new(0));
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

        let res = two_lists_tuple_operation(&f, &g, Operator::Add, fill_value);

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

        let res = scalar_operation(&poly_coeffs, Operator::Mul, scalar);
        assert_eq!(res, poly_coeffs);
    }
}
