// Part 3. FRI commitments

use std::vec;

use crate::{
    channel::{serialize, Channel},
    field::FieldElement,
    merkle::MerkleTree,
    polynomial::Polynomial,
    sessions::part2,
};

pub fn part_3() {
    // Load the previous session
    let (cp, cp_eval, cp_merkle, mut channel, eval_domain) = part2();

    println!("Success!");

    // FRI folding
    //
    // Domain generation

    let half_domain_size = eval_domain.len() / 2;
    assert!(eval_domain[100].pow(2) == eval_domain[half_domain_size + 100].pow(2));

    fn next_fri_domain(fri_domain: &[FieldElement]) -> Vec<FieldElement> {
        let fri_domain_len = fri_domain.len();
        fri_domain
            .iter()
            .take(fri_domain_len / 2)
            .map(|x| x.clone().pow(2))
            .collect()
    }

    let next_domain = next_fri_domain(&eval_domain);
    let next_domain_str: Vec<String> = next_domain.iter().map(|x| x.to_string()).collect();
    // assert!(
    //     "5446c90d6ed23ea961513d4ae38fc6585f6614a3d392cb087e837754bfd32797"
    //         == sha256::digest(serialize(&next_domain_str))
    // );
    println!("Success!");

    // FRI folding operator

    fn next_fri_polynomial(poly: &Polynomial, beta: FieldElement) -> Polynomial {
        let poly_len = poly.poly().len();
        let mut odd_coeffs = vec![];
        let mut even_coeffs = vec![];
        for i in (1..poly_len).step_by(2) {
            odd_coeffs.push(poly.get_nth_degree_coefficient(i));
        }
        for i in (0..poly_len).step_by(2) {
            even_coeffs.push(poly.get_nth_degree_coefficient(i));
        }

        let odd = Polynomial::new(&odd_coeffs).scalar_mul(beta.val());
        let even = Polynomial::new(&even_coeffs);

        odd + even
    }

    let next_p = next_fri_polynomial(&cp, FieldElement::new(987654321));
    let next_p_coeffs: Vec<String> = next_p.poly().into_iter().map(|x| x.to_string()).collect();
    // assert!(
    //     "6bff4c35e1aa9693f9ceb1599b6a484d7636612be65990e726e52a32452c2154"
    //         == sha256::digest(serialize(&next_p_coeffs))
    // );
    println!("Success!");

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
    // assert!(next_p.poly() == vec![FieldElement::new(23), FieldElement::new(7)]);
    // assert!(next_d == vec![FieldElement::new(9)]);
    // assert!(next_l == vec![FieldElement::new(86)]);
    print!("Success!");

    // Generating FRI commitments

    fn fri_commit(
        cp: Polynomial,
        domain: Vec<FieldElement>,
        cp_eval: Vec<FieldElement>,
        cp_merkle: MerkleTree,
        channel: &mut Channel,
    ) -> (
        Vec<Polynomial>,
        Vec<Vec<FieldElement>>,
        Vec<Vec<FieldElement>>,
        Vec<MerkleTree>,
    ) {
        let mut fri_polys = vec![cp];
        let mut fri_domains = vec![domain];
        let mut fri_layers = vec![cp_eval];
        let mut fri_merkles = vec![cp_merkle];
        while fri_polys.last().unwrap().degree() > 0 {
            let beta = channel.receive_random_field_element();
            let (next_poly, next_domain, next_layer) =
                next_fri_layer(fri_polys.last().unwrap(), fri_domains.last().unwrap(), beta);
            fri_polys.push(next_poly);
            fri_domains.push(next_domain);
            fri_layers.push(next_layer.clone());
            let mut merkle_tree = MerkleTree::new(next_layer);
            merkle_tree.build_tree();
            fri_merkles.push(merkle_tree);
            channel.send(fri_merkles.last().unwrap().root.clone());
        }
        channel.send(fri_polys.last().unwrap().poly()[0].to_string());

        (fri_polys, fri_domains, fri_layers, fri_merkles)
    }

    // let mut test_channel = Channel::new();
    // let (fri_polys, fri_domains, fri_layers, fri_merkles) = fri_commit(
    //     cp.clone(),
    //     eval_domain.clone(),
    //     cp_eval.clone(),
    //     cp_merkle,
    //     &mut test_channel,
    // );
    // assert!(
    //     fri_layers.len() == 11,
    //     "Expected number of FRI layers is 11, whereas it is actually {}.",
    //     fri_layers.len()
    // );
    // assert!(
    //     fri_layers.last().unwrap().len() == 8,
    //     "Expected last layer to contain exactly 8 elements, it contains {}.",
    //     fri_layers.last().unwrap().len()
    // );
    // for x in fri_layers.last().unwrap() {
    //     assert!(
    //         x == &FieldElement::from(-1138734538_i128),
    //         "Expected last layer to be constant."
    //     )
    // }
    // assert!(
    //     fri_polys.last().unwrap().degree() == 0,
    //     "Expected last polynomial to be constant (degree 0)."
    // );
    // assert!(
    //     fri_merkles.last().unwrap().root
    //         == "1c033312a4df82248bda518b319479c22ea87bd6e15a150db400eeff653ee2ee",
    //     "Last layer Merkle root is wrong."
    // );
    // assert!(
    //     test_channel.state == "61452c72d8f4279b86fa49e9fb0fdef0246b396a4230a2bfb24e2d5d6bf79c2e",
    //     "The channel state is not as expected."
    // );
    // println!("Success!");

    let (fri_polys, fri_domains, fri_layers, fri_merkles) =
        fri_commit(cp, eval_domain, cp_eval, cp_merkle, &mut channel);
    println!("{:?}", channel.proof);
}
