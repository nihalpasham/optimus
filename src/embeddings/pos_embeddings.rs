use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

/// Holds per word position embeddings.
///
/// According the paper, this need not be a learnable parameter. Its a fixed quantity i.e for
/// each position in a sentence, we can have a position embedding that is to be added
/// to the word embedding.
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
        let denom = ((Tensor::arange_step(0f32, d_model as f32, 2f32, device)?
            * (-10000.0f64.ln() / d_model as f64))?)
            .exp()?;
        let pos = pos.unsqueeze(1)?;
        let denom = denom.unsqueeze(0)?;
        let tmp = (pos.matmul(&denom))?; // produces a matrix with dimensions seq_len * d_model
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
    pub fn forward(&self, ts: Tensor) -> Result<Tensor> {
        let res = (&self.pos_embeddings + ts)?;
        self.dropout.forward(&res, false)
    }
}
