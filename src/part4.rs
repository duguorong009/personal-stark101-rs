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
            let sub_idx = (idx + length / 2) % length;
            channel.send(layer[idx].to_string());
            channel.send(merkle.get_authentication_path(idx));
            channel.send(layer[sub_idx].to_string());
            channel.send(merkle.get_authentication_path(sub_idx));
        }
        let last_layer = fri_layers.last().unwrap();
        channel.send(last_layer[0].to_string());
    };

    let mut test_channel = Channel::new();
    for query in [7527, 8168, 1190, 2668, 1262, 1889, 3828, 5798, 396, 2518] {
        decommit_on_fri_layers(query, &mut test_channel);
    }
    assert!(
        test_channel.state == "ad4fe9aaee0fbbad0130ae0fda896393b879c5078bf57d6c705ec41ce240861b",
        "State of channel is wrong."
    );
    println!("Success!");

    // Decommit on the Trace Polynomial
    let decommit_on_query = |idx: usize, channel: &mut Channel| {
        assert!(
            idx + 16 < f_eval.len(),
            "query index: {} is out of range. Length of layer: {}",
            idx,
            f_eval.len()
        );
        channel.send(f_eval[idx].to_string());
        channel.send(f_merkle.get_authentication_path(idx));
        channel.send(f_eval[idx + 8].to_string());
        channel.send(f_merkle.get_authentication_path(idx + 8));
        channel.send(f_eval[idx + 16]);
        channel.send(f_merkle.get_authentication_path(idx + 16));
        decommit_on_fri_layers(idx, channel);
    };

    let mut test_channel = Channel::new();
    for query in [8134, 1110, 1134, 6106, 7149, 4796, 144, 4738, 957] {
        decommit_on_query(query, &mut test_channel);
    }
    assert!(
        test_channel.state == "16a72acce8d10ffb318f8f5cd557930e38cdba236a40439c9cf04aaf650cfb96",
        "State of channel is wrong"
    );
    println!("Success!");

    // Decommit on a set of queries
    let decommit_fri = |channel: &mut Channel| {
        for query in (0..3) {
            // Get a random index from the verifier and send the corresponding decommitment
            decommit_on_query(channel.receive_random_int(0, 8191 - 16, false), channel);
        }
    };

    let mut test_channel = Channel::new();
    decommit_fri(&mut test_channel);
    assert!(
        test_channel.state == "eb96b3b77fe6cd48cfb388467c72440bdf035c51d0cfe8b4c003dd1e65e952fd",
        "State of channel is wrong."
    );
    println!("Success!");

    // Proving Time!
    use std::time::Instant;
    let before = Instant::now();

    println!("Generating trace...");

    let (_, _, _, _, _, _, _, f_eval, f_merkle, _) = part1();
    println!("{:?}", before.elapsed());

    println!("Generating the composition polynomial and the FRI layers...");
    let (fri_polys, fri_domains, fri_layers, fri_merkles, mut channel) = part3();
    println!("{:?}", before.elapsed());

    println!("Generating the queries and decommitments...");
    decommit_fri(&mut channel);
    println!("{:?}", before.elapsed());

    println!("{}", channel.proof);

    println!("Overall time: {:?}", before.elapsed());
    println!(
        "Uncompressed proof length in characters: {}",
        channel.proof.len()
    );
}
