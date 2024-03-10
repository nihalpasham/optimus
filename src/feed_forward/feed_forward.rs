use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, Dropout, Linear, VarBuilder, VarMap};

/// Represents the FeedForwardBlock in the transformer architecture.
pub struct FeedForwardBlock {
    linear_1: Linear,
    dropout: Dropout,
    linear_2: Linear,
}

impl FeedForwardBlock {
    /// Creates an instance of a new `FeedForwardBlock`. We use a `VarMap` to initialize two linear layers.
    /// A VarMap allows us to initialize tensors using a config (configs here refers to a distribution,
    /// ex: uniform distribution). In this case, we're using the Kaiming distribution. See [`Init`]
    /// for more details
    ///
    /// `linear()` is a helper function that returns the `Linear` type which contains weights and biases.  
    ///
    /// Note:
    /// According to the paper, the 2 linear layers have the following weights and biases
    /// W1 - [512 x 2048], and B1 [512]
    /// W2 - [2048 x 512], and B2 [2048]
    pub fn new(d_model: usize, dropout: f32, d_ff: usize, device: &Device) -> Result<Self> {
        let w1b1 = VarMap::new();
        let vb_w1b1 = VarBuilder::from_varmap(&w1b1, DType::F32, device);

        let w2b2 = VarMap::new();
        let vb_w2b2 = VarBuilder::from_varmap(&w2b2, DType::F32, device);

        let linear_1 = linear(d_model, d_ff, vb_w1b1)?;
        let dropout = Dropout::new(dropout);
        let linear_2 = linear(d_ff, d_model, vb_w2b2)?;

        Ok(Self {
            linear_1,
            dropout,
            linear_2,
        })
    } 

    /// Applying the FeedForwardBlock simply performs the following transformation
    /// (Batch, Seq_Len, d_model) --> (Batch, Seq_Len, d_ff) --> (Batch, Seq_Len, d_model)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.linear_2.forward(
            &self
                .dropout
                .forward(&self.linear_1.forward(x)?.relu()?, true)?,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn verify_feedforward_new() {
        let device = Device::new_metal(0).unwrap();
        let ff = FeedForwardBlock::new(512, 0.3, 2048, &device).unwrap();
        println!("linear_1: {}\n", ff.linear_1.weight());
        println!("linear_1: {}\n", ff.linear_1.bias().unwrap());
        println!("linear_2: {}\n", ff.linear_2.weight());
        println!("linear_2: {}\n", ff.linear_2.bias().unwrap());
    }
}