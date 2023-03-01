use crate::{
    channel::Channel,
    field::FieldElement,
    merkle::MerkleTree,
    polynomial::{prod, Polynomial, X},
    sessions::part1,
};

pub fn part_2() {
    // Step 1. FibonacciSq Constraints

    // Step 2. Polynomial Constraints
    let (a, g, G, h, H, eval_domain, f, f_eval, f_merkle, channel) = part1();

    // Step 3. Rational Functions(That are in Fact Polynomials)

    // The first constraint
    let numer_0 = f.clone() - 1;
    let denom_0 = X() - 1;
    assert!(f.eval(FieldElement::new(1)) == 1);

    let p_0 = numer_0 / denom_0;
    assert!(p_0.eval(FieldElement::new(2718)) == 2509888982);

    println!("Success!");

    // The second constraint
    let numer_1 = f - 2338775057;
    let denom_1 = X() - g.pow(1022);
    let p_1 = numer_1 / denom_1;

    assert!(p_1.eval(FieldElement::new(5772)) == 232961446);

    println!("Success!");

    // The third constraint - succinctness
    let lst: Vec<Polynomial> = (0..1024).map(|i| X() - g.pow(i)).collect();
    prod(&lst);

    // Composing Polynomials(a detour)
    let q = X().pow(2) * 2_usize + 1;
    let r = X() - 3;
    // let q_r = q * r;
    let q_r = Polynomial::new(&[]); // should be q(r)
    assert!(
        q_r == Polynomial::new(&[
            FieldElement::from(19_i128),
            FieldElement::from(-12_i128),
            FieldElement::from(2_i128)
        ])
    );
    println!("Success!");

    // Back to Polynomial constraints
    let numer_2 = Polynomial::new(&[]); // Should be numer2 = f(g**2 * X) - f(g * X)**2 - f**2
    let denom_2 =
        (X().pow(1024) - 1) / ((X() - g.pow(1021)) * (X() - g.pow(1022)) * (X() - g.pow(1023)));
    let p_2 = numer_2 / denom_2;

    assert!(
        p_2.degree() == 1023,
        "The degree of third constraint is {}, when it should be 1023",
        p_2.degree()
    );
    assert!(p_2.eval(FieldElement::new(31415)) == 2090051528);
    println!("Success!");

    // Step 4. Composition Polynomial
    let get_CP = |channel: &mut Channel| -> Polynomial {
        let a_0 = channel.receive_random_field_element();
        let a_1 = channel.receive_random_field_element();
        let a_2 = channel.receive_random_field_element();

        p_0.clone() * a_0 + p_1.clone() * a_1 + p_2.clone() * a_2
    };

    let mut test_channel = Channel::new();
    let CP_test = get_CP(&mut test_channel);

    assert!(
        CP_test.degree() == 1023,
        "The degree of cp is {}, when it should be 1023.",
        CP_test.degree()
    );
    assert!(CP_test.eval(FieldElement::new(2439804)) == 838767343);

    println!("Success!");

    // Commit on the Composition polynomial
    let CP_eval = |channel: &mut Channel| -> Vec<FieldElement> {
        let CP = get_CP(channel);
        eval_domain.iter().map(|d| CP.eval(*d)).collect()
    };

    let mut channel = Channel::new();
    let mut CP_merkle = MerkleTree::new(CP_eval(&mut channel));
    CP_merkle.build_tree();

    channel.send(CP_merkle.root.clone());

    assert!(
        CP_merkle.root == "a8c87ef9764af3fa005a1a2cf3ec8db50e754ccb655be7597ead15ed4a9110f1",
        "Merkle tree root is wrong."
    );
    println!("Success!");
}
