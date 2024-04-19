use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

use crate::{layer_norm::norm::LayerNormalization, utils::IsResidualLayerInput};

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
        mask: Option<Tensor>,
        sublayer: impl IsResidualLayerInput,
    ) -> Result<Tensor> {
        let tmp = self.norm.forward(xs)?;
        let sublayer_tensor = sublayer.forward(&tmp, mask)?;
        // apply dropout and combine the original tensor
        let res = xs + self.dropout.forward(&sublayer_tensor, false)?;
        res
    }
}
