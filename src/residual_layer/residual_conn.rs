use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock, layer_norm::norm::LayerNormalization,
    multi_head_attn::multihead_block::MultiHeadAttnBlock, utils::IsResidualLayerInput,
};

pub enum SubLayers<'a> {
    Mha(&'a MultiHeadAttnBlock),
    Ff(&'a FeedForwardBlock),
}
#[derive(Debug)]
pub struct ResidualConnection {
    dropout: Dropout,
    norm: LayerNormalization,
}

impl ResidualConnection {
    pub fn new(dropout: f32) -> Result<Self> {
        let dropout = Dropout::new(dropout);
        let norm = LayerNormalization::new()?;
        Ok(ResidualConnection { dropout, norm })
    }

    /// Runs the skip connection.
    pub fn forward(
        &self,
        xs: &Tensor,
        xa: Option<&Tensor>,
        mask: Option<Tensor>,
        sublayer: SubLayers,
    ) -> Result<Tensor> {
        let t = self.norm.forward(xs)?;
        let sublayer_tensor = match sublayer {
            SubLayers::Mha(m) => match xa {
                Some(xa) => {
                    let xa = self.norm.forward(xa)?;
                    m.forward(&t, &xa, &xa, mask)?
                }
                None => m.forward(&t, &t, &t, mask)?,
            },
            SubLayers::Ff(f) => f.forward(&t)?,
        };
        // apply dropout and combine the original tensor
        let res = xs + self.dropout.forward(&sublayer_tensor, false)?;
        res
    }
}
