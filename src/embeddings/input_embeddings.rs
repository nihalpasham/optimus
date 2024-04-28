use candle_core::{DType, Module};
use std::collections::HashMap;

use candle_core::{Device, Result, Tensor};
use candle_nn::{embedding, Embedding, VarBuilder};

/// Holds vector embeddings for each token along with the vector dimensions and
/// vocabulary length
#[derive(Debug)]
pub struct InputEmbeddings {
    #[allow(dead_code)]
    d_model: usize,
    #[allow(dead_code)]
    vocab_size: usize,
    embedding: Embedding,
}

impl InputEmbeddings {
    /// Creates an instance of a new `InputEmbedding`. The underlying storage (i.e. embedding matrix)
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

    /// Creates an embedding (i.e. embedding vectors for each word) when provided with a sequence of
    /// token_ids or indices
    pub fn forward(&self, indices: &[u32], device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_slice(indices, (indices.len(),), device)?;
        // the paper performs a scalar multiplication of the embedding matrix with square root of d_model
        let dmodel_sqrt = (self.d_model as f32).sqrt();
        let t = Tensor::new(dmodel_sqrt, device)?;
        self.embedding.forward(&tensor)?.broadcast_mul(&t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use tokenizers::Tokenizer;

    #[test]
    fn verify_input_embeddings_new() {
        // load a pre-trained tokenizer
        let tokenizer = Tokenizer::from_file("./src/tokenizer/wordlevel-wiki.json").unwrap();
        let encoding = tokenizer
            .encode(("Welcome to the library. ", "test this out"), true)
            .unwrap();
        println!("tok:  {:?}", encoding.get_tokens());
        // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
        println!("ids:  {:?}", encoding.get_ids());
        // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]

        let vocab_size = tokenizer.get_vocab_size(true);
        let token_ids = encoding.get_ids();

        let device = Device::new_metal(0).unwrap();
        let input_embeddings = InputEmbeddings::new(vocab_size, 512, &device).unwrap();

        let embeddings = input_embeddings.forward(&token_ids, &device).unwrap();

        println!("vector embeddings:\n {}\n", embeddings);
    }
}
