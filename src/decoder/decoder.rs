use super::decoder_block::DecoderBlock;
use crate::layer_norm::norm::LayerNormalization;
use candle_core::{Result, Tensor};

/// The `Decoder` represents the decoder component of the Transformer architecture.
/// It consists of a series of `DecoderBlock` layers, followed by a layer normalization step.
///
/// The decoder takes in the previous decoder outputs and the encoder output, and produces
/// the next decoder output by applying multi-head self-attention, multi-head cross-attention,
/// and feed-forward neural network layers in each `DecoderBlock`. The self-attention allows
/// the decoder to attend to the previous outputs, while the cross-attention allows it to
/// attend to the encoder output, enabling the decoder to incorporate information from the
/// input sequence.
///
/// The final decoder output is produced by passing the output of the last `DecoderBlock`
/// through a layer normalization module.
///
/// # Example
///
/// ```rust
/// use transformer::decoder::Decoder;
/// use transformer::decoder_block::DecoderBlock;
///
/// // Create the decoder blocks
/// let decoder_blocks: Vec<DecoderBlock> = /* ... */;
///
/// // Create the decoder
/// let decoder = Decoder::new(decoder_blocks).unwrap();
///
/// // Perform the forward pass
/// let decoder_output = decoder.forward(
///     prev_decoder_output,
///     &encoder_output,
///     true, // Apply target mask
///     false, // Don't apply source mask
/// ).unwrap();
/// ```
#[derive(Debug)]
pub struct Decoder<'a> {
    /// The vector of `DecoderBlock` layers that make up the decoder.
    layers: Vec<DecoderBlock<'a>>,
    /// The layer normalization module applied after the decoder blocks.
    norm: LayerNormalization,
}

impl<'a> Decoder<'a> {
    // Creates a new `Decoder` instance with the given `DecoderBlock` layers.
    ///
    /// # Arguments
    ///
    /// * `layers` - The vector of `DecoderBlock` layers.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `Decoder` instance on success, or an error if
    /// the layer normalization module could not be created.
    pub fn new(layers: Vec<DecoderBlock<'a>>) -> Result<Self> {
        let norm = LayerNormalization::new()?;
        Ok(Decoder { layers, norm })
    }
    /// Performs the forward pass through the decoder component.
    ///
    /// # Arguments
    ///
    /// * `xs` - The input tensor representing the previous decoder outputs.
    /// * `encoder_output` - The output tensor from the encoder part of the Transformer.
    /// * `tgt_msk` - A boolean indicating whether to apply a mask to the previous decoder outputs.
    /// * `src_mask` - A boolean indicating whether to apply a mask to the encoder output.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the output tensor after applying the decoder blocks and
    /// layer normalization, or an error if any of the decoder block operations failed.
    pub fn forward(
        &self,
        mut xs: Tensor,
        encoder_ouput: &Tensor,
        tgt_msk: bool,
        src_mask: bool,
    ) -> Result<Tensor> {
        for blk in self.layers.iter() {
            xs = blk.forward(&xs, encoder_ouput, src_mask, tgt_msk)?
        }
        self.norm.forward(&xs)
    }
}

#[cfg(test)]
mod tests {
    use core::num;

    use candle_core::Device;
    use candle_nn::Dropout;
    use tokenizers::Tokenizer;

    use crate::{
        embeddings::{input_embeddings::InputEmbeddings, pos_embeddings::PosEmbeddings},
        feed_forward::feed_forward::FeedForwardBlock,
        multi_head_attn::multihead_block::MultiHeadAttnBlock,
    };

    use super::*;

    #[test]
    fn test_decoder() {
        let d_model = 512usize;
        let d_ff = 2048usize;
        let num_heads = 4usize;
        let dropout = 0.3;

        let device = Device::Cpu;
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
        let seq_len = encoding.get_ids().len();

        let input_embeds = InputEmbeddings::new(vocab_size, d_model, &device).unwrap();
        let embeddings = input_embeds.forward(&token_ids, &device).unwrap();
        println!("vector embeddings: \n{}\n", embeddings);
        let mut pe = PosEmbeddings::new(seq_len, d_model, Dropout::new(dropout), &device).unwrap();
        println!("pos_embeddings main: \n{}\n", pe.pos_embeddings);
        let decoder_input = pe.forward(embeddings).unwrap();
        println!("Decoder_input: \n{}\n", decoder_input);

        let mut layers = Vec::with_capacity(10);
        for layer in 0..layers.capacity() {
            layers.push(
                DecoderBlock::new(
                    MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device).unwrap(),
                    MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device).unwrap(),
                    FeedForwardBlock::new(d_model, dropout, d_ff, &device).unwrap(),
                    dropout,
                )
                .unwrap(),
            )
        }
        let dummy_encoder_output = decoder_input.clone();

        let decoder = Decoder::new(layers).unwrap();
        let t = decoder
            .forward(decoder_input, &dummy_encoder_output, false, true)
            .unwrap();
        println!("Decoder_output: \n{}\n", t);
    }
}
