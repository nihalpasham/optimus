use candle_core::{DType, Device, Module, Result, Storage, Tensor, Var};
use candle_nn::{linear, ops::log_softmax, Linear, VarBuilder, VarMap};

#[derive(Debug, Clone)]
pub struct ProjectionLayer(Linear);

impl ProjectionLayer {
    pub fn new(d_model: usize, vocab_size: usize) -> Result<Self> {
        let backend = VarMap::new();
        let storage = VarBuilder::from_varmap(&backend, DType::F32, &Device::Cpu);
        let linear_layer = linear(d_model, vocab_size, storage)?;
        Ok(ProjectionLayer(linear_layer))
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let p = self.0.forward(xs)?;
        let last_dim = p.dims().len();
        log_softmax(&p, last_dim)
    }
}
