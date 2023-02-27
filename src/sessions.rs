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

pub fn part2() {
    let (t, g, points, h_gen, h, domain, p, ev, mt, ch) = part1();

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

    //

    todo!()
}
