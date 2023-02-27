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

    let points_usize = points.iter().map(|x| x.val()).collect();
    let t_usize = t.iter().map(|x| x.val()).collect();
    let p = interpolate_poly(points_usize, t_usize);

    let ev: Vec<FieldElement> = domain.iter().map(|d| p.eval(d.clone())).collect();

    let mut mt = MerkleTree::new(ev.clone());
    mt.build_tree();

    let mut ch = Channel::new();
    ch.send(mt.root.clone());

    (t, g, points, h_gen, h, domain, p, ev, mt, ch)
}
