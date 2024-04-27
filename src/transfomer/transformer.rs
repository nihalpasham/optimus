use candle_core::{Device, Result, Tensor};
use candle_nn::Dropout;

use crate::{
    decoder::{decoder::Decoder, decoder_block::DecoderBlock},
    embeddings::{input_embeddings::InputEmbeddings, pos_embeddings::PosEmbeddings},
    encoder::{encoder::Encoder, encoder_block::EncoderBlock},
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    projection_layer::projection::ProjectionLayer,
};

#[derive(Debug)]
pub struct Transformer {
    enc_embed: InputEmbeddings,
    dec_embed: InputEmbeddings,
    enc_pos: PosEmbeddings,
    dec_pos: PosEmbeddings,
    encoder: Encoder,
    decoder: Decoder,
    projection_layer: ProjectionLayer,
}

impl Transformer {
    pub fn new(
        enc_embed: InputEmbeddings,
        dec_embed: InputEmbeddings,
        enc_pos: PosEmbeddings,
        dec_pos: PosEmbeddings,
        encoder: Encoder,
        decoder: Decoder,
        projection_layer: ProjectionLayer,
    ) -> Self {
        Transformer {
            enc_embed,
            dec_embed,
            enc_pos,
            dec_pos,
            encoder,
            decoder,
            projection_layer,
        }
    }
    pub fn encode(
        &mut self,
        indices: &[u32],
        src_mask: Option<Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let src_embeddings = self.enc_embed.forward(indices, &device)?;
        let src_embeddings = self.enc_pos.forward(src_embeddings)?;
        self.encoder.forward(src_embeddings, src_mask)
    }
    pub fn decode(
        &mut self,
        indices: &[u32],
        encoder_input: &Tensor,
        src_mask: Option<Tensor>,
        tgt_mask: Option<Tensor>,
        device: &Device,
    ) -> Result<Tensor> {
        let tgt_embeddings = self.dec_embed.forward(indices, &device)?;
        let tgt_embeddings = self.dec_pos.forward(tgt_embeddings)?;
        self.decoder
            .forward(tgt_embeddings, encoder_input, tgt_mask, src_mask)
    }
    pub fn project(&self, input: &Tensor) -> Result<Tensor> {
        self.projection_layer.forward(input)
    }
}

pub fn build_transformer(
    src_vocab_size: usize,
    tgt_vocab_size: usize,
    src_seq_len: usize,
    tgt_seq_len: usize,
) -> Result<Transformer> {
    let hidden_size = 6usize;
    let d_model = 512usize;
    let d_ff = 2048usize;
    let num_heads = 8usize;
    let dropout = 0.1;

    let device = Device::new_metal(0)?;

    let src_embeds = InputEmbeddings::new(src_vocab_size, d_model, &device)?;
    let tgt_embeds = InputEmbeddings::new(tgt_vocab_size, d_model, &device)?;

    let mut src_pos = PosEmbeddings::new(src_seq_len, d_model, Dropout::new(dropout), &device)?;
    let mut tgt_pos = PosEmbeddings::new(tgt_seq_len, d_model, Dropout::new(dropout), &device)?;

    let mut encoder_blocks = Vec::with_capacity(hidden_size);
    for layer in 0..encoder_blocks.capacity() {
        encoder_blocks.push(EncoderBlock::new(
            MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device)?,
            FeedForwardBlock::new(d_model, dropout, d_ff, &device)?,
            dropout,
        )?)
    }

    let mut decoder_blocks = Vec::with_capacity(hidden_size);
    for layer in 0..decoder_blocks.capacity() {
        decoder_blocks.push(DecoderBlock::new(
            MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device)?,
            MultiHeadAttnBlock::new(d_model, num_heads, dropout, &device)?,
            FeedForwardBlock::new(d_model, dropout, d_ff, &device)?,
            dropout,
        )?)
    }

    let encoder = Encoder::new(encoder_blocks)?;
    let decoder = Decoder::new(decoder_blocks)?;

    let projection_layer = ProjectionLayer::new(d_model, tgt_vocab_size)?;

    Ok(Transformer {
        enc_embed: src_embeds,
        dec_embed: tgt_embeds,
        enc_pos: src_pos,
        dec_pos: tgt_pos,
        encoder,
        decoder,
        projection_layer,
    })
}
