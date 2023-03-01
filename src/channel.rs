use num::{BigInt, Num, ToPrimitive};

use crate::field::FieldElement;

/// Serializes an object into a string.
pub fn serialize(obj: &[String]) -> String {
    obj.join(",")
}

// A Channel instance can be used by a prover or a verifier to preserve the semantics of an
// interactive proof system, while under the hood it is in fact non-interactive, and uses Sha256
// to generate randomness when this is required.
// It allows writing string-form data to it, and reading either random integers of random
// FieldElements from it.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Channel {
    pub state: String,
    pub proof: Vec<String>,
}

impl Channel {
    pub fn new() -> Self {
        Self {
            state: "0".to_string(),
            proof: vec![],
        }
    }

    pub fn send(&mut self, s: String) {
        let input = format!("{}{}", self.state.clone(), s);
        let digest = sha256::digest(input);

        self.state = digest;
        self.proof.push(format!("send:{}", s));
    }

    /// Emulates a random integer sent by the verifier in the range [min, max] (including min and
    /// max).
    ///
    /// "show_in_proof: true(default)"
    pub fn receive_random_int(&mut self, min: usize, max: usize, show_in_proof: bool) -> usize {
        // Note that when the range is close to 2^256 this does not emit a uniform distribution,
        // even if sha256 is uniformly distributed.
        // It is, however, close enough for this tutorial's purposes.
        let num = min
            + (BigInt::from_str_radix(&self.state, 16).unwrap() % (max - min + 1))
                .to_usize()
                .unwrap();

        let digest = sha256::digest(self.state.clone());
        self.state = digest;

        if show_in_proof {
            self.proof.push(format!("receive_random_int:{}", num));
        }

        num
    }

    ///  Emulates a random field element sent by the verifier.
    pub fn receive_random_field_element(&mut self) -> FieldElement {
        let num = self.receive_random_int(0, FieldElement::k_modulus() - 1, false);
        self.proof
            .push(format!("receive_random_field_element:{}", num));

        FieldElement::new(num)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reproducability() {
        let mut c = Channel::new();
        c.send("Yes".to_string());
        let r1 = c.receive_random_int(0, 2_usize.pow(20), true);

        let mut d = Channel::new();
        d.send("Yes".to_string());
        let r2 = d.receive_random_int(0, 2_usize.pow(20), true);

        assert!(r1 == r2);
    }

    #[test]
    fn test_uniformity() {
        // TODO
        assert!(1 + 1 == 2);
    }
}
