// Part 3. FRI commitments

use crate::{channel::serialize, field::FieldElement, polynomial::Polynomial, sessions::part2};

pub fn part_3() {
    // Load the previous session
    // let (cp, cp_eval, cp_merkle, channel, eval_domain) = part2();

    println!("Success!");

    // FRI folding
    //
    // Domain generation

    // let half_domain_size = eval_domain.len() / 2;
    // assert!(eval_domain[100].pow(2) == eval_domain[half_domain_size + 100].pow(2));

    fn next_fri_domain(fri_domain: &[FieldElement]) -> Vec<FieldElement> {
        let fri_domain_len = fri_domain.len();
        fri_domain
            .iter()
            .take(fri_domain_len / 2)
            .map(|x| x.clone().pow(2))
            .collect()
    }

    // let next_domain = next_fri_domain(&eval_domain);
    // let next_domain_str: Vec<String> = next_domain.iter().map(|x| x.to_string()).collect();
    // assert!(
    //     "5446c90d6ed23ea961513d4ae38fc6585f6614a3d392cb087e837754bfd32797"
    //         == sha256::digest(serialize(&next_domain_str))
    // );
    // println!("Success!");

    // FRI folding operator

    fn next_fri_polynomial(poly: &Polynomial, beta: FieldElement) -> Polynomial {
        let poly_len = poly.poly().len();
        let mut odd_coeffs = vec![];
        let mut even_coeffs = vec![];
        for i in (1..poly_len).step_by(2) {
            odd_coeffs.push(poly.get_nth_degree_coefficient(i as usize));
        }
        for i in (0..poly_len).step_by(2) {
            even_coeffs.push(poly.get_nth_degree_coefficient(i as usize));
        }

        let odd = Polynomial::new(&odd_coeffs).scalar_mul(beta.val());
        let even = Polynomial::new(&even_coeffs);

        odd + even
    }

    // let next_p = next_fri_polynomial(&cp, FieldElement::new(987654321));
    // let next_p_coeffs: Vec<String> = next_p.poly().into_iter().map(|x| x.to_string()).collect();
    // assert!(
    //     "6bff4c35e1aa9693f9ceb1599b6a484d7636612be65990e726e52a32452c2154"
    //         == sha256::digest(serialize(&next_p_coeffs))
    // );
    // println!("Success!");

    // Putting it together to get the next FRI layer
    fn next_fri_layer(
        poly: &Polynomial,
        dom: &[FieldElement],
        alpha: FieldElement,
    ) -> (Polynomial, Vec<FieldElement>, Vec<FieldElement>) {
        let next_poly = next_fri_polynomial(poly, alpha);
        let next_dom = next_fri_domain(dom);
        let next_layer = next_dom.iter().map(|x| next_poly.eval(*x)).collect();

        (next_poly, next_dom, next_layer)
    }

    let test_poly = Polynomial::new(&[
        FieldElement::new(2),
        FieldElement::new(3),
        FieldElement::new(0),
        FieldElement::new(1),
    ]);
    let test_domain = vec![FieldElement::new(3), FieldElement::new(5)];
    let beta = FieldElement::new(7);

    let (next_p, next_d, next_l) = next_fri_layer(&test_poly, &test_domain, beta);
    assert!(next_p.poly() == vec![FieldElement::new(23), FieldElement::new(7)]);
    assert!(next_d == vec![FieldElement::new(9)]);
    assert!(next_l == vec![FieldElement::new(86)]);
    print!("Success!");

    // Generating FRI commitments
}
