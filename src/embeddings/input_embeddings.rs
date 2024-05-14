use candle_core::{
    op::{Op, ReduceOp, UnaryOp},
    DType, Module, TensorId,
};

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
    fn sort_nodes(&self) -> Vec<&Tensor>;
    fn get_op_graph(sorted_nodes: Vec<&Tensor>);
}
impl SortedNodes for Tensor {
    /// Return all the nodes that lead to this value in a topologically sorted vec, the first
    /// elements having dependencies on the latter ones, e.g. the first element if any is the
    /// argument.
    /// This assumes that the op graph is a DAG.
    fn sort_nodes(&self) -> Vec<&Tensor> {
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
        let (_tg, mut nodes) = walk(self, vec![], &mut HashMap::new());
        nodes.reverse();
        nodes
    }

    fn get_op_graph(sorted_nodes: Vec<&Tensor>) {
        for node in sorted_nodes.iter() {
            let op = match node.op() {
                Some(op) => match op {
                    candle_core::op::Op::Binary(a, b, c) => {
                        println!("BinaryOp: {:?}, arg1: {:?}, arg2: {:?}", c, a.id(), b.id())
                    }
                    candle_core::op::Op::Unary(a, b) => {
                        println!("UnaryOp: {:?}, arg: {:?}", b, a.id())
                    }
                    candle_core::op::Op::Cmp(a, b) => {
                        println!("CmpOp: {:?}, arg: {:?}, ", b, a.id())
                    }
                    candle_core::op::Op::Reduce(a, b, c) => {
                        println!("ReduceOp: {:?}, arg: {:?}, output: {:?}", b, a.id(), c)
                    }
                    candle_core::op::Op::Matmul(a, b) => {
                        println!("MatmulOp: arg1: {:?}, arg2: {:?}", a.id(), b.id())
                    }
                    candle_core::op::Op::Gather(a, b, c) => {
                        println!(
                            "GatherOp: arg1: {:?}, arg2: {:?}, output: {:?}",
                            a.id(),
                            b.id(),
                            c
                        )
                    }
                    candle_core::op::Op::ScatterAdd(a, b, c, d) => println!(
                        "ScatterAddOp: arg1: {:?}, arg2: {:?}, arg3: {:?}, output: {:?}",
                        a.id(),
                        b.id(),
                        c.id(),
                        d
                    ),
                    candle_core::op::Op::IndexSelect(a, b, c) => println!(
                        "IndexSelectOp: arg1: {:?}, arg2: {:?}, output: {:?}",
                        a.id(),
                        b.id(),
                        c
                    ),
                    candle_core::op::Op::IndexAdd(a, b, c, d) => println!(
                        "IndexAddOp: arg1: {:?}, arg2: {:?}, arg3: {:?}, output: {:?}",
                        a.id(),
                        b.id(),
                        c.id(),
                        d
                    ),
                    candle_core::op::Op::WhereCond(a, b, c) => println!(
                        "WhereCondOp: arg1: {:?}, arg2: {:?}, arg3: {:?}",
                        a.id(),
                        b.id(),
                        c.id()
                    ),
                    candle_core::op::Op::Conv1D {
                        arg,
                        kernel,
                        padding,
                        stride,
                        dilation,
                    } => println!("Conv1DOp:"),
                    candle_core::op::Op::ConvTranspose1D {
                        arg,
                        kernel,
                        padding,
                        output_padding,
                        stride,
                        dilation,
                    } => println!("ConvTranspose1DOp:"),
                    candle_core::op::Op::Conv2D {
                        arg,
                        kernel,
                        padding,
                        stride,
                        dilation,
                    } => println!("Conv2DOp:"),
                    candle_core::op::Op::ConvTranspose2D {
                        arg,
                        kernel,
                        padding,
                        output_padding,
                        stride,
                        dilation,
                    } => println!("ConvTranspose2DOp:"),
                    candle_core::op::Op::AvgPool2D {
                        arg,
                        kernel_size,
                        stride,
                    } => println!("AvgPool2DOp:"),
                    candle_core::op::Op::MaxPool2D {
                        arg,
                        kernel_size,
                        stride,
                    } => println!("MaxPool2DOp:"),
                    candle_core::op::Op::UpsampleNearest1D { arg, target_size } => {
                        println!("UpsampleNearest1DOp:")
                    }
                    candle_core::op::Op::UpsampleNearest2D {
                        arg,
                        target_h,
                        target_w,
                    } => println!("UpsampleNearest2DOp:"),
                    candle_core::op::Op::Cat(a, b) => println!("CatOp"),
                    candle_core::op::Op::Affine { arg, mul, add } => {
                        println!("AffineOp: {} Mul, {} Add", mul, add)
                    }
                    candle_core::op::Op::ToDType(a) => println!("ToDtypeOp"),
                    candle_core::op::Op::Copy(a) => println!("CopyOp: arg: {:?}", a.id()),
                    candle_core::op::Op::Broadcast(a) => println!("BroadcastOp: arg: {:?}", a.id()),
                    candle_core::op::Op::Narrow(a, b, c, d) => println!("NarrowOp:"),
                    candle_core::op::Op::SliceScatter0(a, b, c) => println!("SliceScatter0Op:"),
                    candle_core::op::Op::Reshape(a) => println!("ReshapeOp: arg: {:?}", a.id()),
                    candle_core::op::Op::ToDevice(a) => println!("ToDeviceOp"),
                    candle_core::op::Op::Transpose(a, b, c) => {
                        println!("TransposeOp: ({} , {}) dims, arg: {:?}", b, c, a.id())
                    }
                    candle_core::op::Op::Permute(a, b) => println!("PermuteOp:"),
                    candle_core::op::Op::Elu(a, b) => println!("EluOp:"),
                    candle_core::op::Op::Powf(a, b) => println!("PowfOp:"),
                    candle_core::op::Op::CustomOp1(a, b) => println!("CustomOp1:"),
                    candle_core::op::Op::CustomOp2(a, b, c) => println!("CustomOp2:"),
                    candle_core::op::Op::CustomOp3(a, b, c, d) => println!("CustomOp3:"),
                },
                None => println!("None: arg: {:?}", node.id()),
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::{linear, VarMap};
    use tokenizers::Tokenizer;

    #[test]
    fn test_metal_kernel_launch() {
        let bsize = 1usize;
        let x = 8usize;
        let y = 512usize;
        let device = Device::new_metal(0).unwrap();
        let vmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, &device);
        let ll = linear(512, 512, vb.pp("ll")).unwrap();

        let random = Tensor::randn(0f32, 1., (bsize, x, y), &device).unwrap();
        let res = ll.forward(&random).unwrap();
        let ordered_nodes = res.sort_nodes();
        Tensor::get_op_graph(ordered_nodes);
        let (b_size, seq_len, _) = random.dims3().unwrap();
        let res = res
            .reshape((b_size, seq_len, 4, 512 / 4))
            .unwrap()
            .transpose(1, 2)
            .unwrap();
        let ordered_nodes = res.sort_nodes();
        println!("\n");
        Tensor::get_op_graph(ordered_nodes);
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
        let sorted_nodes = embeddings.sort_nodes();
        println!("sorted_nodes: \n{:?}\n", sorted_nodes);
    }
}
