use crate::{
    channel::Channel,
    sessions::{part1, part3},
};

/// Part 5. Query Phase
pub fn part_4() {
    // Load the previous session
    let (_, _, _, _, _, _, _, f_eval, f_merkle, _) = part1();

    let (fri_polys, fri_domains, fri_layers, fri_merkles, _) = part3();

    println!("Success!");

    // Decommit on a query

    // Decommit on the FRI layers
    let decommit_on_fri_layers = |idx: usize, channel: &mut Channel| {
        for i in 0..(fri_layers.len() - 1) {
            let layer = &fri_layers[i];
            let merkle = &fri_merkles[i];

            let length = layer.len();
            let idx = idx % length;
            let sib_idx = (idx + length / 2) % length;
            channel.send(layer[idx].to_string());
            channel.send(merkle.get_authentication_path(idx));
            channel.send(layer[sub_idx].to_string());
            channel.send(merkle.get_authentication_path(sub_idx));
        }
        let last_layer = fri_layers.last().unwrap();
        channel.send(last_layer[0].to_string());
    };

    // Decommit on the Trace Polynomial

    // Decommit on a set of queries

    // Proving Time!
}
