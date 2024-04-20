use candle_core::{Device, Module, Result, Tensor};
use candle_nn::LayerNorm;

/// In deep neural networks, especially with many layers, the  outputs of neurons can grow very
/// large or very small over time. This can make training difficult. Layer normalization, used in the "Attention
/// is All You Need" paper, helps address this issue.
///
/// Imagine each layer's output as a set of values. Layer normalization calculates the
/// average and standard deviation of these values.
///
/// It then subtracts the average from each value and divides it by the standard deviation. This effectively
/// "standardizes" the outputs, bringing them closer to a normal distribution with a mean of 0 and standard
/// deviation of 1.
#[derive(Debug, Clone)]
pub struct LayerNormalization(LayerNorm);

impl LayerNormalization {
    /// `LayerNormalizartion` wraps the built-in `LayerNorm` type.
    pub fn new() -> Result<Self> {
        let w = Tensor::new(1f32, &Device::Cpu)?;
        let b = Tensor::new(0f32, &Device::Cpu)?;
        let layer_norm = LayerNorm::new(w, b, 1e-5);
        Ok(LayerNormalization(layer_norm))
    }

    /// Performs the layer normalization.
    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.0.forward(xs)
    }
}
