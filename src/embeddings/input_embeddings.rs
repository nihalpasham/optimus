use candle_core::{DType, Module, TensorId, op::{Op, UnaryOp, ReduceOp}};

use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

/// Holds vector embeddings for each token along with the vector dimensions and
/// vocabulary length
#[derive(Debug)]
pub struct InputEmbeddings {
    #[allow(dead_code)]
    d_model: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    embedding: Embedding,
}

impl InputEmbeddings {
    /// Creates an instance of a new `InputEmbedding`. The underlying storage (i.e. embedding matrix)
    /// starts of as a `Tensor` initialized with random elements of type `f32`.
    ///
    /// Note: the tensor's dimensions are (vocab_size, d_model) and elements are ranged with
    ///     - `mean:` 0.0 and
    ///     - `std dev:` 1.0
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        map.insert(
            String::from("weight"),
            Tensor::randn(0f32, 1., (vocab_size, d_model), &device)?,
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        let embedding = embedding(vocab_size, d_model, vb)?;

        Ok(InputEmbeddings {
            d_model,
            vocab_size,
            embedding,
        })
    }

    /// Creates an embedding (i.e. embedding vectors for each word) when provided with a sequence of
    /// token_ids or indices
    pub fn forward(&self, indices: &[u32], device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;
        // the paper performs a scalar multiplication of the embedding matrix with square root of d_model
        let dmodel_sqrt = (self.d_model as f32).sqrt();
        let t = Tensor::new(dmodel_sqrt, device)?;
        self.embedding.forward(&tensor)?.broadcast_mul(&t)
    }
}

pub trait SortedNodes {
    fn new_sorted_nodes(&self) -> Vec<&Tensor> ;
}
impl SortedNodes for Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn new_sorted_nodes(&self) -> Vec<&Tensor> {
        // The vec of sorted nodes is passed as an owned value rather than a mutable reference
        // to get around some lifetime limitations.
        fn walk<'a>(
            node: &'a Tensor,
            nodes: Vec<&'a Tensor>,
            already_seen: &mut HashMap<TensorId, bool>,
        ) -> (bool, Vec<&'a Tensor>) {
            if let Some(&tg) = already_seen.get(&node.id()) {
                return (tg, nodes);
            }
            let mut track_grad = false;
            let mut nodes = if node.is_variable() {
                // Do not call recursively on the "leaf" nodes.
                track_grad = true;
                // println!("is_variable");
                nodes
            } else if node.dtype().is_int() {
                // println!("is_init");
                nodes
            } else if let Some(op) = node.op() {
                match op {
                    Op::IndexAdd(t1, t2, t3, _)
                    | Op::ScatterAdd(t1, t2, t3, _)
                    | Op::CustomOp3(t1, t2, t3, _)
                    | Op::WhereCond(t1, t2, t3) => {
                        let (tg, nodes) = walk(t1, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t2, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(t3, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Conv1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose1D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::Conv2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::ConvTranspose2D {
                        arg: lhs,
                        kernel: rhs,
                        ..
                    }
                    | Op::CustomOp2(lhs, rhs, _)
                    | Op::Binary(lhs, rhs, _)
                    | Op::Gather(lhs, rhs, _)
                    | Op::IndexSelect(lhs, rhs, _)
                    | Op::Matmul(lhs, rhs)
                    | Op::SliceScatter0(lhs, rhs, _) => {
                        let (tg, nodes) = walk(lhs, nodes, already_seen);
                        track_grad |= tg;
                        let (tg, nodes) = walk(rhs, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::Cat(args, _) => args.iter().fold(nodes, |nodes, arg| {
                        let (tg, nodes) = walk(arg, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }),
                    Op::Affine { arg, mul, .. } => {
                        if *mul == 0. {
                            nodes
                        } else {
                            let (tg, nodes) = walk(arg, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                    }
                    Op::Unary(_node, UnaryOp::Ceil)
                    | Op::Unary(_node, UnaryOp::Floor)
                    | Op::Unary(_node, UnaryOp::Round)
                    | Op::Unary(_node, UnaryOp::Sign) => nodes,
                    Op::Reshape(node)
                    | Op::UpsampleNearest1D { arg: node, .. }
                    | Op::UpsampleNearest2D { arg: node, .. }
                    | Op::AvgPool2D { arg: node, .. }
                    | Op::MaxPool2D { arg: node, .. }
                    | Op::Copy(node)
                    | Op::Broadcast(node)
                    | Op::Cmp(node, _)
                    | Op::Reduce(node, ReduceOp::Min | ReduceOp::Sum | ReduceOp::Max, _)
                    | Op::ToDevice(node)
                    | Op::Transpose(node, _, _)
                    | Op::Permute(node, _)
                    | Op::Narrow(node, _, _, _)
                    | Op::Unary(node, _)
                    | Op::Elu(node, _)
                    | Op::Powf(node, _)
                    | Op::CustomOp1(node, _) => {
                        let (tg, nodes) = walk(node, nodes, already_seen);
                        track_grad |= tg;
                        nodes
                    }
                    Op::ToDType(node) => {
                        if node.dtype().is_float() {
                            let (tg, nodes) = walk(node, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        } else {
                            nodes
                        }
                    }
                    Op::Reduce(_, ReduceOp::ArgMin | ReduceOp::ArgMax, _) => nodes,
                }
            } else {
                nodes
            };
            already_seen.insert(node.id(), track_grad);
            if track_grad {
                nodes.push(node);
            }
            (track_grad, nodes)
        }
        println!("\n in sorted nodes");
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use tokenizers::Tokenizer;

    #[test]
    fn test_metal_kernel_launch() {
        let x = 2048usize;
        let y = 512usize;
        let device = Device::new_metal(0).unwrap();
        let t = Tensor::arange(0., 1., &device).unwrap();
        let v1 = Tensor::randn(0f32, 1., (x, y), &device).unwrap();
        let v2 = Tensor::randn(0f32, 1., (x, y), &device).unwrap();

        let v2 = v1.matmul(&v2).unwrap();
        let v3 = (&v1 + &v2).unwrap();
        let v4 = (&v3 - &v2).unwrap();
        let v5 = (&v4 * &v2).unwrap();
        let v6 = (v5.tanh()).unwrap();
        
        let tp = v6.new_sorted_nodes();
        println!("topological order: {:?}", tp);
        println!("f: {}", v6);
    }

    #[test]
    fn verify_input_embeddings_new() {
        // load a pre-trained tokenizer
        let tokenizer = Tokenizer::from_file("./src/tokenizer/wordlevel-wiki.json").unwrap();
        let encoding = tokenizer
            .encode(("Welcome to the library. ", "test this out"), true)
            .unwrap();
        println!("tok:  {:?}", encoding.get_tokens());
        // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
        println!("ids:  {:?}", encoding.get_ids());
        // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]

        let vocab_size = tokenizer.get_vocab_size(true);
        let token_ids = encoding.get_ids();

        let device = Device::new_metal(0).unwrap();
        let input_embeddings = InputEmbeddings::new(vocab_size, 512, &device).unwrap();

        let embeddings = input_embeddings.forward(&token_ids, &device).unwrap();

        println!("vector embeddings:\n {}\n", embeddings);
        let sorted_nodes  = embeddings.new_sorted_nodes();
        println!("sorted_nodes: \n{:?}\n", sorted_nodes);
    }
}
