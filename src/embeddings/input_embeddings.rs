use candle_core::{DType, Module};
use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

/// Holds vector embeddings for each token along with the vector dimensions and
/// vocabulary length
pub struct InputEmbeddings {
    #[allow(dead_code)]
    d_model: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    embedding: Embedding,
}

impl InputEmbeddings {
    /// Creates a new `InputEmbedding` instance. The underlying storage (i.e. embedding matrix)
    /// starts of as a `Tensor` initialized with random elements of type `f32`.
    ///
    /// Note: the tensor's dimensions are (vocab_size, d_model) and elements are ranged with
    ///     - `mean:` 0.0 and
    ///     - `std dev:` 1.0
    pub fn new(vocab_size: usize, d_model: usize, device: &Device) -> Result<Self> {
        let mut map = HashMap::new();
        map.insert(
            String::from("weight"),
            Tensor::randn(0f32, 1., (vocab_size, d_model), &device)?,
        );
        let vb = VarBuilder::from_tensors(map, DType::F32, device);
        let embedding = embedding(vocab_size, d_model, vb)?;

        Ok(InputEmbeddings {
            d_model,
            vocab_size,
            embedding,
        })
    }

    /// Creates an embedding (i.e. embedding vectors for each word) from a sequence of supplied 
    /// token_ids or indices 
    pub fn forward(&self, indices: &[u32], device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;
        self.embedding.forward(&tensor)
    }
}
