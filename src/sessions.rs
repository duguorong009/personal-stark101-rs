use crate::channel::Channel;
use crate::field::FieldElement;
use crate::merkle::MerkleTree;
use crate::polynomial::{interpolate_poly, Polynomial};

pub fn part1() -> (
    Vec<FieldElement>,
    FieldElement,
    Vec<FieldElement>,
    FieldElement,
    Vec<FieldElement>,
    Vec<FieldElement>,
    Polynomial,
    Vec<FieldElement>,
    MerkleTree,
    Channel,
) {
    let mut t = vec![FieldElement::new(1), FieldElement::new(3141592)];
    while t.len() < 1023 {
        let second_last = t[t.len() - 2];
        let last = t[t.len() - 1];
        t.push(second_last * second_last + last * last);
    }

    let g = FieldElement::generator().pow(3 * 2_usize.pow(20));
    let points: Vec<FieldElement> = (0..1024).map(|i| g.pow(i)).collect();
    let h_gen = FieldElement::generator().pow((2_usize.pow(30) * 3) / 8192);
    let h: Vec<FieldElement> = (0..8192).map(|i| h_gen.pow(i)).collect();

    let domain: Vec<FieldElement> = h
        .iter()
        .map(|x| FieldElement::generator() * x.clone())
        .collect();

    let mut points_usize: Vec<usize> = points.iter().map(|x| x.val()).collect();
    points_usize.pop();
    let t_usize = t.iter().map(|x| x.val()).collect();
    let p = interpolate_poly(points_usize, t_usize);

    let ev: Vec<FieldElement> = domain.iter().map(|d| p.eval(d.clone())).collect();

    let mut mt = MerkleTree::new(ev.clone());
    mt.build_tree();

    let mut ch = Channel::new();
    ch.send(mt.root.clone());

    (t, g, points, h_gen, h, domain, p, ev, mt, ch)
}

pub fn part2() -> (
    Polynomial,
    Vec<FieldElement>,
    MerkleTree,
    Channel,
    Vec<FieldElement>,
) {
    let (t, g, points, h_gen, h, domain, p, ev, mt, mut ch) = part1();

    let numer_0 = p.clone() - Polynomial::new(&[FieldElement::one()]);
    let denom_0 = Polynomial::gen_linear_term(FieldElement::one());
    let (q_0, r_0) = numer_0.qdiv(denom_0);

    let numer_1 = p.clone() - Polynomial::new(&[FieldElement::new(2338775057)]);
    let denom_1 = Polynomial::gen_linear_term(points[1022]);
    let (q_1, r_1) = numer_1.qdiv(denom_1);

    let inner_poly_0 = Polynomial::new(&[FieldElement::zero(), points[2]]);
    let final_0 = p.compose(inner_poly_0);

    let inner_poly_1 = Polynomial::new(&[FieldElement::zero(), points[1]]);
    let composition = p.compose(inner_poly_1);

    let final_1 = composition.clone() * composition;
    let final_2 = p.clone() * p.clone();

    let numer_2 = final_0 - final_1 - final_2;
    let mut coef = vec![FieldElement::zero(); 1025];
    coef[0] = FieldElement::new(1);
    coef[1024] = FieldElement::from(-1_i128);
    let numerator_of_denom_2 = Polynomial::new(&coef);

    let factor_0 = Polynomial::gen_linear_term(points[1021]);
    let factor_1 = Polynomial::gen_linear_term(points[1022]);
    let factor_2 = Polynomial::gen_linear_term(points[1023]);

    let denom_of_denom_2 = factor_0 * factor_1 * factor_2;

    let (denom_2, r_denom_2) = numerator_of_denom_2.qdiv(denom_of_denom_2);

    let (q_2, r_2) = numer_2.qdiv(denom_2);

    let cp_0 = q_0.scalar_mul(ch.receive_random_field_field_element().val());
    let cp_1 = q_1.scalar_mul(ch.receive_random_field_field_element().val());
    let cp_2 = q_2.scalar_mul(ch.receive_random_field_field_element().val());

    let cp = cp_0 + cp_1 + cp_2;

    let cp_ev: Vec<FieldElement> = domain.iter().map(|d| cp.eval(d.clone())).collect();

    let mut cp_mt = MerkleTree::new(cp_ev.clone());
    cp_mt.build_tree();

    ch.send(cp_mt.root.clone());

    (cp, cp_ev, cp_mt, ch, domain)
}
