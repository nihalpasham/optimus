use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

pub struct PosEmbeddings {
    pos_embeddings: Tensor,
    seq_len: usize,
    d_model: usize,
    dropout: Dropout,
}

impl PosEmbeddings {
    pub fn new(seq_len: usize, d_model: usize, dropout: Dropout, device: &Device) -> Result<Self> {
        let pos = Tensor::arange(0f32, seq_len as f32, device)?;
        let denom = ((Tensor::arange_step(0f32, d_model as f32, 2f32, device)?
            * (-10000.0f64.ln() / d_model as f64))?)
            .exp()?;
        let pos = pos.unsqueeze(1)?;
        let denom = denom.unsqueeze(0)?;
        let tmp = (pos.matmul(&denom))?;
        let even_embeds = tmp.sin()?;
        let odd_embeds = tmp.cos()?;

        let even_col0 = even_embeds.get_on_dim(1, 0)?;
        let odd_col0 = odd_embeds.get_on_dim(1, 0)?;
        // println!("even_col0: {}\n, odd_col0: {}", even_col0, odd_col0);

        let mut pos_embeddings = Tensor::cat(&[&even_col0, &odd_col0], 0)?;

        for col in 1..d_model / 2 {
            let even_col = even_embeds.get_on_dim(1, col)?;
            pos_embeddings = Tensor::cat(&[&pos_embeddings, &even_col], 0)?;
            let odd_col = odd_embeds.get_on_dim(1, col)?;
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

    pub fn forward(&self, ts: Tensor) -> Result<Tensor> {
        let res = (&self.pos_embeddings + ts)?;
        self.dropout.forward(&res, false)
    }
}
