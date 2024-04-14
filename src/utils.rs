use candle_core::{Device, Result, Tensor};

pub trait IsResidualLayerInput {
    fn forward(&self, x: &Tensor, mask: Option<Tensor>) -> Result<Tensor>;
}
