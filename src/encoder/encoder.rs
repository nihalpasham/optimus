use super::encoder_block::EncoderBlock;
use crate::layer_norm::norm::LayerNormalization;
use candle_core::{Result, Tensor};

pub struct Encoder<const N: usize> {
    enc_blks: [EncoderBlock; N],
    norm: LayerNormalization,
}

impl<const N: usize> Encoder<N> {
    pub fn new(enc_blks: [EncoderBlock; N]) -> Result<Self> {
        let norm = LayerNormalization::new()?;
        Ok(Encoder { enc_blks, norm })
    }
    pub fn forward(&self, mut xs: Tensor, src_mask: Option<Tensor>) -> Result<Tensor> {
        for blk in self.enc_blks.iter() {
            xs = blk.forward(&xs, src_mask.clone())?
        }
        self.norm.forward(&xs)
    }
}
