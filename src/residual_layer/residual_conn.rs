use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

use crate::{layer_norm::norm::LayerNormalization, utils::IsResidualLayerInput};

pub struct ResidualConnection {
    dropout: Dropout,
    norm: LayerNormalization,
}

impl ResidualConnection {
    pub fn new(device: &Device, dropout: f32) -> Result<Self> {
        let dropout = Dropout::new(dropout);
        let norm = LayerNormalization::new(device)?;
        Ok(ResidualConnection { dropout, norm })
    }

    /// Runs the skip connection.
    pub fn forward(
        &self,
        xs: &Tensor,
        mask: Tensor,
        sublayer: impl IsResidualLayerInput,
    ) -> Result<Tensor> {
        let tmp = self.norm.forward(xs)?;
        let sublayer_tensor = sublayer.forward(&tmp, Some(mask))?;
        // apply dropout and combine the original tensor
        let res = xs + self.dropout.forward(&sublayer_tensor, false)?;
        res
    }
}