use std::collections::HashMap;

use crate::field::FieldElement;

enum MerkleTreeNode {
    Internal { left: String, right: String },
    Leaf { leaf_data: String },
}

/// Merkle tree implementation
/// This implementation is not generic, but specific to
/// this tutorial.
///
/// Usage:
/// let mut mt = MerkleTree(ev);
/// mt.build_tree();
pub struct MerkleTree {
    height: u32,
    data: Vec<FieldElement>,
    facts: HashMap<String, MerkleTreeNode>,
    pub root: String,
}

impl MerkleTree {
    pub fn new(leaves: Vec<FieldElement>) -> Self {
        assert!(!leaves.is_empty(), "Cannot construct an empty Merkle Tree");

        let height = (leaves.len() as f32).log2().ceil() as u32;
        let num_leaves = 2_usize.pow(height);

        let mut data = leaves.clone();
        for _ in 0..(num_leaves - leaves.len()) {
            data.push(FieldElement::zero());
        }

        Self {
            height,
            data,
            root: "".to_string(),
            facts: HashMap::new(),
        }
    }

    pub fn build_tree(&mut self) -> String {
        let root = self.recursive_build_tree(1);

        self.root = root.clone();

        root
    }

    fn recursive_build_tree(&mut self, node_id: usize) -> String {
        if node_id >= self.data.len() {
            // A leaf
            let id_in_data = node_id - self.data.len();
            let leaf_data = self.data[id_in_data].to_string();
            let h = sha256::digest(leaf_data.clone());
            self.facts
                .insert(h.clone(), MerkleTreeNode::Leaf { leaf_data });
            h
        } else {
            // An internal node
            let left = self.recursive_build_tree(node_id * 2);
            let right = self.recursive_build_tree(node_id * 2 + 1);
            let data = format!("{}{}", left, right);
            let h = sha256::digest(data);
            self.facts
                .insert(h.clone(), MerkleTreeNode::Internal { left, right });
            h
        }
    }

    pub fn get_authentication_path(&self, leaf_id: usize) -> Vec<String> {
        assert!(leaf_id < self.data.len());

        let node_id = leaf_id + self.data.len();
        let mut cur = self.root.clone();

        let mut decommitment: Vec<String> = vec![];

        // In a Merkle Tree, the path from the root to a leaf, corresponds to the the leaf id's
        // binary representation, starting from the second-MSB, where '0' means 'left', and '1' means
        // 'right'.
        // We therefore iterate over the bits of the binary representation - skipping the '0b'
        // prefix, as well as the MSB.
        let bits = format!("{:b}", node_id).chars().take(3).collect::<String>();

        for bit in bits.chars() {
            let mut auth: String = "".to_string();

            let merkle_node = self.facts.get(&cur).unwrap();
            (cur, auth) = match merkle_node {
                MerkleTreeNode::Internal { left, right } => (left.to_string(), right.to_string()),
                MerkleTreeNode::Leaf { leaf_data: _ } => panic!("Cannot be leaf!"),
            };

            if bit.to_string() == "1" {
                (auth, cur) = (cur, auth);
            }
            decommitment.push(auth);
        }

        decommitment
    }
}

pub fn verify_decommitment(
    leaf_id: usize,
    leaf_data: FieldElement,
    decommitment: &[String],
    root: String,
) -> bool {
    let leaf_num = 2_usize.pow(decommitment.len() as u32);

    let node_id = leaf_id + leaf_num;

    let mut cur = sha256::digest(leaf_data.to_string());

    let bits = format!("{:b}", node_id)
        .chars()
        .rev()
        .take(3)
        .collect::<String>();

    for (bit, auth) in bits.chars().zip(decommitment.iter().rev()) {
        let h = if bit.to_string() == *"0" {
            format!("{}{}", cur, auth)
        } else {
            format!("{}{}", auth, cur)
        };
        cur = sha256::digest(h);
    }

    cur == root
}
