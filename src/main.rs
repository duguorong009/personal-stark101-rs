use field::FieldElement;

mod channel;
mod field;
mod list_utils;
mod merkle;
mod polynomial;

fn main() {
    println!("Hello, world!");
}

fn part1() {
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
    // TODO
    // let p = interpolate_poly(points, t);
}
