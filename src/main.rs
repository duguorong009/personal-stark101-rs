use crate::{
    channel::{serialize, Channel},
    field::FieldElement,
    merkle::MerkleTree,
    polynomial::{interpolate_poly, X},
    sessions::{part1, part2, part3},
};

mod channel;
mod field;
mod list_utils;
mod merkle;
mod polynomial;

mod sessions;

fn main() {
    ///////////////////  Part 1 ////////////////////////

    // The Basics
    // FieldElement Class
    assert!(FieldElement::new(3221225472) + FieldElement::new(10) == FieldElement::new(9));

    // FibonacciSq Trace
    let mut a = vec![FieldElement::new(1), FieldElement::new(3141592)];
    while a.len() < 1023 {
        let second_last = a[a.len() - 2];
        let last = a[a.len() - 1];
        a.push(second_last * second_last + last * last);
    }

    assert!(
        a.len() == 1023,
        "The trace must consist of exactly 1023 elements"
    );
    assert!(
        a[0] == FieldElement::one(),
        "The first element in the trace must be the unit element"
    );
    for i in (2..1023) {
        assert!(
            a[i] == a[i - 1] * a[i - 1] + a[i - 2] * a[i - 2],
            "The FibonacciSq recursion rule does not apply for index {i}"
        );
    }
    assert!(
        a[1022] == FieldElement::new(2338775057),
        "Wrong last element"
    );
    println!("Success!");

    // Thinking of Polynomials
    //
    // Find a Group of Size 1024
    let g = FieldElement::generator().pow(3 * 2_usize.pow(20));
    let G: Vec<FieldElement> = (0..1024).map(|i| g.pow(i)).collect();

    assert!(g.is_order(1024), "The generator g is of wrong order");
    let mut b = FieldElement::one();
    for i in 0..1023 {
        assert!(
            b == G[i],
            "The i-th place on G is not equal to the i-th power of g."
        );
        b *= g;
        assert!(b != FieldElement::one(), "g is of order {}", i + 1);
    }
    if b * g == FieldElement::one() {
        println!("Success!");
    } else {
        println!("g is of order > 1024");
    }

    // // INFO
    // // Polynomial Class
    // let p = X().pow(2) * 2_usize + 1;
    // println!("p: {:?}", p);

    // Interpolating a Polynomial
    let f = interpolate_poly(&G[0..G.len() - 1], &a);
    let v = f.eval(FieldElement::new(2));
    assert!(v == FieldElement::new(1302089273));
    println!("Success!");

    // Evaluating on a Larger Domain
    //
    // Cosets
    let h = FieldElement::generator().pow(3 * 2_usize.pow(17));
    let H: Vec<FieldElement> = (0..8192).map(|i| h.pow(i)).collect();

    let w = FieldElement::generator();
    let eval_domain: Vec<FieldElement> = H.iter().map(|x| w * *x).collect();

    let w_inv = w.inverse();
    assert!(
        H[1] == h,
        "H list is incorrect. H[1] should be h(i.e., the generator of H)."
    );
    for i in 0..8192 {
        assert!((w_inv * eval_domain[1]).pow(i) * w == eval_domain[i]);
    }
    println!("Success!");

    // Evaluate on a Coset
    let f_eval: Vec<FieldElement> = eval_domain.iter().map(|d| f.eval(*d)).collect();
    let f_eval_str: Vec<String> = f_eval.iter().map(|x| x.to_string()).collect();
    assert!(
        "1d357f674c27194715d1440f6a166e30855550cb8cb8efeb72827f6a1bf9b5bb"
            == sha256::digest(serialize(&f_eval_str))
    );
    println!("Success!");

    // Commitments
    let mut f_merkle = MerkleTree::new(f_eval);
    f_merkle.build_tree();

    assert!(f_merkle.root == "6c266a104eeaceae93c14ad799ce595ec8c2764359d7ad1b4b7c57a4da52be04");
    println!("Success!");

    // Channel
    let mut channel = Channel::new();
    channel.send(f_merkle.root);

    println!("{:?}", channel.proof);
    // ['send:6c266a104eeaceae93c14ad799ce595ec8c2764359d7ad1b4b7c57a4da52be04']

    ////////////////////////////////////////////////////
}
