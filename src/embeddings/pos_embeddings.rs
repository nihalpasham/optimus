use candle_core::{Device, IndexOp, Result, Tensor};
use candle_nn::Dropout;

/// Holds per word position embeddings.
///
/// According the paper, this need not be a learnable parameter. Its can be a fixed quantity i.e for
/// each position in a sentence, we can have a position embedding that is to be added
/// to the word embedding.
#[derive(Debug)]
pub struct PosEmbeddings {
    pub pos_embeddings: Tensor,
    seq_len: usize,
    d_model: usize,
    dropout: Dropout,
}

impl PosEmbeddings {
    /// Construct a new position embedding
    ///
    /// We simply compute position embeddings with the formula used in the paper
    /// - For each position, the even rows are given - P E(pos,2i) = sin(pos/10000^(2i/d_model) )
    /// - For each position, the odd rows are given - P E(pos,2i+1) = cos(pos/10000^(2i/d_model) )
    pub fn new(seq_len: usize, d_model: usize, dropout: Dropout, device: &Device) -> Result<Self> {
        let pos = Tensor::arange(0f32, seq_len as f32, device)?;
        let denom = ((Tensor::arange(0f32, d_model as f32, device)?
            * (-10000.0f64.ln() / d_model as f64))?)
            .exp()?;
        // expand tensor on dim 1 (i.e. column dimension), transforms [pos ] --> [pos, 1]
        let pos = pos.unsqueeze(1)?;
        // expand tensor on dim 0 (i.e. row dimension), transforms [denom ] --> [1, denom]
        let denom = denom.unsqueeze(0)?;
        let tmp = (pos.matmul(&denom))?; // produces a matrix with dimensions seq_len * d_model
        println!("tmp: {}\n", tmp);
        let even_embeds = tmp.sin()?; // apply a sine op to each element in the matrix
        let odd_embeds = tmp.cos()?; // apply a cosine op to each element in the matrix

        let even_col0 = even_embeds.get_on_dim(1, 0)?; // get the first even column
        let odd_col0 = odd_embeds.get_on_dim(1, 1)?; // get the first odd column
                                                     // println!("even_col0: {}\n, odd_col0: {}", even_col0, odd_col0);

        // concatenate the two cols along dimension 0
        let mut pos_embeddings = Tensor::cat(&[&even_col0, &odd_col0], 0)?;

        // iterate over d_model length/2 and keep concatenating even-odd columns
        for col in 1..d_model / 2 {
            let even_col = even_embeds.get_on_dim(1, col * 2)?;
            pos_embeddings = Tensor::cat(&[&pos_embeddings, &even_col], 0)?;
            let odd_col = odd_embeds.get_on_dim(1, col * 2 + 1)?;
            pos_embeddings = Tensor::cat(&[&pos_embeddings, &odd_col], 0)?;
        }
        // produces a shape of [1, Seq_Len, d_model]
        pos_embeddings = pos_embeddings
            .reshape((d_model, seq_len))?
            .transpose(0, 1)?
            .unsqueeze(0)?;

        Ok(PosEmbeddings {
            pos_embeddings,
            seq_len,
            d_model,
            dropout,
        })
    }

    /// Add `position embeddings` for each word to each `word embedding` to incorporate position
    /// information into our input.
    ///
    /// Note: This implementation only supports a single batch of input tokens
    pub fn forward(&mut self, ts: Tensor) -> Result<Tensor> {
        let res = (&self.pos_embeddings.i(0)? + ts)?;
        self.pos_embeddings = res.unsqueeze(0)?;
        self.dropout.forward(&self.pos_embeddings, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn verify_pos_embeddings_new() {
        let device = Device::new_metal(0).unwrap();
        let pe = PosEmbeddings::new(8, 512, Dropout::new(0.3), &device).unwrap();
        println!("positional embeddings: {}\n", pe.pos_embeddings);
    }
}
