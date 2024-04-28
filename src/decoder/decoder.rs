use super::decoder_block::DecoderBlock;
use crate::layer_norm::norm::LayerNormalization;
use candle_core::{Result, Tensor};

#[derive(Debug)]
pub struct Decoder<'a> {
    layers: Vec<DecoderBlock<'a>>,
    norm: LayerNormalization,
}

impl<'a> Decoder<'a> {
    pub fn new(layers: Vec<DecoderBlock<'a>>) -> Result<Self> {
        let norm = LayerNormalization::new()?;
        Ok(Decoder { layers, norm })
    }
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
