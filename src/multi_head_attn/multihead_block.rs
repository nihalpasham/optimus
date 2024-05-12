use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{
    init, linear,
    ops::{log_softmax, softmax},
    Dropout, Init, Linear, VarBuilder, VarMap,
};

/// Represents the `Multi-Head Attention Block` in the transformer architecture.
#[derive(Debug)]
pub struct MultiHeadAttnBlock<'a> {
    d_model: usize,
    /// number of num_heads
    num_heads: usize,
    /// each head's dimension size
    head_size: usize,
    dropout: Dropout,
    /// Wq matrix
    w_q: Linear,
    /// Wk matrix
    w_k: Linear,
    /// Wv matrix
    w_v: Linear,
    /// Wo - output weight matrix
    w_o: Linear,
    /// hold a ref to the underlying device - for storage
    device: &'a Device,
}

impl<'a> MultiHeadAttnBlock<'a> {
    /// Creates an instance of a new `MultiHeadAttnBlock`. We use a `VarMap` to initialize 4 linear layers.
    /// A VarMap allows us to initialize tensors using a config (configs here refers to a distribution,
    /// ex: uniform distribution). In this case, we're using the Kaiming distribution. See [`Init`]
    /// for more details
    ///
    /// Note:
    /// According to the paper, the 4 linear layers have the following weights and biases
    /// Wq - [512 x 512], and Bq [512]
    /// Wk - [512 x 512], and Bk [512]
    /// Wv - [512 x 512], and Bv [512]
    /// Wo - [512 x 512], and Bo [512]
    pub fn new(d_model: usize, num_heads: usize, dropout: f32, device: &'a Device) -> Result<Self> {
        let vmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&vmap, DType::F32, device);
        let wq = linear(d_model, d_model, vb.pp("wq"))?;
        let wk = linear(d_model, d_model, vb.pp("wk"))?;
        let wv = linear(d_model, d_model, vb.pp("wv"))?;
        let wo = linear(d_model, d_model, vb.pp("wo"))?;

        let dropout = Dropout::new(dropout);
        assert!(d_model % num_heads == 0);

        Ok(Self {
            d_model,
            num_heads,
            head_size: d_model / num_heads,
            dropout,
            w_q: wq,
            w_k: wk,
            w_v: wv,
            w_o: wo,
            device,
        })
    }

    pub fn compute_attn_scores(
        &self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Option<Tensor>,
        bsize: usize,
        num_heads: usize,
        dropout: &Dropout,
    ) -> Result<(Tensor, Tensor)> {
        let head_size = match query.dims().last() {
            Some(v) => v,
            None => {
                let s = query.shape();
                return Err(candle_core::Error::DimOutOfRange {
                    shape: s.clone(),
                    dim: -1,
                    op: "invalid last dim",
                });
            }
        };

        let sqrt = (*head_size as f32).sqrt();
        let t = Tensor::new(sqrt, self.device)?;

        // (Batch, num_heads, Seq_Len, head_size) --> (Batch, num_heads, Seq_Len, Seq_Len)
        let attn_scores = query
            .contiguous()?
            .matmul(&key.t()?.contiguous()?)?
            .broadcast_div(&t)?;

        // println!("raw_attn_scores: \n{}\n", attn_scores);
        // apply mask
        let mut attn_scores = match mask {
            Some(m) => masked_fill(
                &attn_scores,
                &m.broadcast_left((bsize, num_heads))?,
                f32::NEG_INFINITY,
            )?,
            None => attn_scores,
        };
        // println!("masked_attn_scores: \n{}\n", attn_scores);
        let last_dim = attn_scores.dims().len();
        // (Batch, num_heads, Seq_Len, Seq_Len) --> (Batch, num_heads, Seq_Len, Seq_Len)
        attn_scores = softmax(&attn_scores, last_dim - 1)?;
        //apply dropout
        // println!("softmaxed_attn_scores: \n{}\n", attn_scores);
        attn_scores = dropout.forward(&attn_scores, false)?;
        // (Batch, num_heads, Seq_Len, Seq_Len) --> (Batch, num_heads, Seq_Len, head_size)
        let final_attn_scores = attn_scores.contiguous()?.matmul(&value.contiguous()?)?;
        Ok((final_attn_scores, attn_scores))
    }

    /// Applying the `MultiheadAttnBlock` simply performs the following transformation
    pub fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: bool) -> Result<Tensor> {
        let q_prime = self.w_q.forward(q)?; // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        let k_prime = self.w_k.forward(k)?; // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        let v_prime = self.w_v.forward(v)?; // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        // println!("q_prime: {}", q_prime);
        let (b_size, seq_len, _) = q.dims3()?;
        // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, num_heads, head_size) --> (Batch, num_heads, Seq_Len, head_size)
        let query = q_prime
            .reshape((b_size, seq_len, self.num_heads, self.head_size))?
            .transpose(1, 2)?;
        // println!("query: \n{}\n", query);
        // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, num_heads, head_size) --> (Batch, num_heads, Seq_Len, head_size)
        let key = k_prime
            .reshape((b_size, seq_len, self.num_heads, self.head_size))?
            .transpose(1, 2)?;
        // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, num_heads, head_size) --> (Batch, num_heads, Seq_Len, head_size)
        let value = v_prime
            .reshape((b_size, seq_len, self.num_heads, self.head_size))?
            .transpose(1, 2)?;

        // Apply a mask, if provided
        let mask = match mask {
            true => Some(get_mask(seq_len, self.device)?),
            false => None,
        };
        // println!("mask: \n{:?}\n", mask.clone());
        let (attn_scores, raw_attn_scores) = self.compute_attn_scores(
            query,
            key,
            value,
            mask,
            b_size,
            self.num_heads,
            &self.dropout,
        )?;
        // println!("final_attn_scores: {}", attn_scores);
        // (Batch, num_heads, Seq_Len, head_size) --> (Batch, Seq_Len, num_heads, head_size) --> (Batch, Seq_Len, d_model)
        let res = attn_scores.transpose(1, 2)?.contiguous()?.reshape((
            b_size,
            seq_len,
            self.num_heads * self.head_size,
        ))?;

        // (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_model)
        Ok((self.w_o.forward(&res))?)
    }
}

pub fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    // println!("Shape: {:?}", shape);
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    // println!("on_true: {}", on_true);
    let m = mask.where_cond(&on_true, on_false)?;
    // println!("m: {:?}", m);
    Ok(m)
}

#[cfg(test)]
mod tests {
    use candle_core::test_utils::to_vec2_round;
    use tokenizers::Tokenizer;

    use crate::embeddings::{input_embeddings::{InputEmbeddings, SortedNodes}, pos_embeddings::PosEmbeddings};

    use super::*;

    #[test]
    fn verify_multiheadattnblock_new() {
        let device = Device::new_metal(0).unwrap();
        let mha = MultiHeadAttnBlock::new(512, 4, 0.3, &device).unwrap();
        println!("w_q: {}\n", mha.w_q.weight());
        println!("w_q_bias: {}\n", mha.w_q.bias().unwrap());
        println!("w_k: {}\n", mha.w_k.weight());
        println!("w_k_bias: {}\n", mha.w_k.bias().unwrap());
    }

    #[test]
    fn test_get_mask() {
        let device = Device::new_metal(0).unwrap();
        let mask = get_mask(6, &device).unwrap();
        println!("mask: {}", mask);
    }
    #[test]
    fn test_masked_fill() {
        let device = Device::new_metal(0).unwrap();
        let mask = get_mask(6, &device).unwrap();
        println!("mask: {}", mask);
    }

    #[test]
    fn test2() {
        let a = Tensor::new(
            &[[
                1.1389e4, -5.1048e3, -1.6326e2, -9.3767e2, 2.7612e3, 1.3011e2, -7.4047e2, -1.1109e2,
            ]],
            &Device::Cpu,
        )
        .unwrap();
        println!("a.shape(): {}", a);
        let a = candle_nn::ops::softmax(&a, 1).unwrap();
    }
    #[test]
    fn test_multiheadattnblock_forward() {
        let device = Device::new_metal(0).unwrap();

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

        let input_embeds = InputEmbeddings::new(vocab_size, 512, &device).unwrap();
        let embeddings = input_embeds.forward(&token_ids, &device).unwrap();
        println!("vector embeddings: \n{}\n", embeddings);
        let mut pe = PosEmbeddings::new(8, 512, Dropout::new(0.3), &device).unwrap();
        println!("pos_embeddings main: \n{}\n", pe.pos_embeddings);
        let encoder_input = pe.forward(embeddings).unwrap();
        println!("encoder_input: \n{}\n", encoder_input);

        let mha = MultiHeadAttnBlock::new(512, 4, 0.3, &device).unwrap();

        let attn = mha
            .forward(&encoder_input, &encoder_input, &encoder_input, true)
            .unwrap();
        println!("\n attn_output: {}", attn);
        let sorted_nodes  = attn.new_sorted_nodes();
        // println!("sorted_nodes: \n{:?}\n", sorted_nodes);
        for node in sorted_nodes.iter() {
            let op = match node.op() {
                Some(op) => match op {
                    candle_core::op::Op::Binary(_, _, _) => println!("binary op"),
                    candle_core::op::Op::Unary(_, _) => println!("unary op"),
                    candle_core::op::Op::Cmp(_, _) => println!("cmp op"),
                    candle_core::op::Op::Reduce(_, _, _) => println!("reduce op"),
                    candle_core::op::Op::Matmul(_, _) => println!("matmul op"),
                    candle_core::op::Op::Gather(_, _, _) => println!("gather op"),
                    candle_core::op::Op::ScatterAdd(_, _, _, _) => println!("scatterAdd op"),
                    candle_core::op::Op::IndexSelect(_, _, _) => println!("indexselect op"),
                    candle_core::op::Op::IndexAdd(_, _, _, _) => println!("indexadd op"),
                    candle_core::op::Op::WhereCond(_, _, _) => println!("where cond op"),
                    candle_core::op::Op::Conv1D { arg, kernel, padding, stride, dilation } => println!("conv1d op"),
                    candle_core::op::Op::ConvTranspose1D { arg, kernel, padding, output_padding, stride, dilation } => println!("convtranspose1d op"),
                    candle_core::op::Op::Conv2D { arg, kernel, padding, stride, dilation } => println!("conv2d op"),
                    candle_core::op::Op::ConvTranspose2D { arg, kernel, padding, output_padding, stride, dilation } => println!("conv2d transpose op"),
                    candle_core::op::Op::AvgPool2D { arg, kernel_size, stride } => println!("avgpool2d op"),
                    candle_core::op::Op::MaxPool2D { arg, kernel_size, stride } => println!("maxpool2d op"),
                    candle_core::op::Op::UpsampleNearest1D { arg, target_size } => println!("upsamplenearest1d op"),
                    candle_core::op::Op::UpsampleNearest2D { arg, target_h, target_w } => println!("upsamplenearest2d op"),
                    candle_core::op::Op::Cat(_, _) => println!("cat op"),
                    candle_core::op::Op::Affine { arg, mul, add } => println!("affine op"),
                    candle_core::op::Op::ToDType(_) => println!("todtype op"),
                    candle_core::op::Op::Copy(_) => println!("copy op"),
                    candle_core::op::Op::Broadcast(_) => println!("broadcast op"),
                    candle_core::op::Op::Narrow(_, _, _, _) => println!("narrow op"),
                    candle_core::op::Op::SliceScatter0(_, _, _) => println!("slicescatter0 op"),
                    candle_core::op::Op::Reshape(_) => println!("reshape op"),
                    candle_core::op::Op::ToDevice(_) => println!("todevice op"),
                    candle_core::op::Op::Transpose(_, _, _) => println!("transpose op"),
                    candle_core::op::Op::Permute(_, _) => println!("permute op"),
                    candle_core::op::Op::Elu(_, _) => println!("elu op"),
                    candle_core::op::Op::Powf(_, _) => println!("powf op"),
                    candle_core::op::Op::CustomOp1(_, _) => println!("custom1 op"),
                    candle_core::op::Op::CustomOp2(_, _, _) => println!("custom2 op"),
                    candle_core::op::Op::CustomOp3(_, _, _, _) => println!("custom3 op"),
                }
                None => println!("none"),
            };
        }
    }
}
