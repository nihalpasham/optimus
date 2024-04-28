use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::{ResidualConnection, SubLayers},
};

#[derive(Debug)]
pub struct EncoderBlock<'a> {
    mha: MultiHeadAttnBlock<'a>,
    ff: FeedForwardBlock,
    rconns: Vec<ResidualConnection>,
}

impl<'a> EncoderBlock<'a> {
    pub fn new(mha: MultiHeadAttnBlock<'a>, ff: FeedForwardBlock, dropout: f32) -> Result<Self> {
        let mut rconns = Vec::with_capacity(2);
        for conn in 0..2 {
            rconns.push(ResidualConnection::new(dropout)?);
        }
        Ok(EncoderBlock { mha, ff, rconns })
    }

    pub fn forward(&self, xs: &Tensor, src_mask: bool) -> Result<Tensor> {
        let x = self.rconns[0].forward(xs, None, src_mask, SubLayers::Mha(&self.mha))?;
        let x = self.rconns[1].forward(&x, None, false, SubLayers::Ff(&self.ff))?;
        Ok(x)
    }
}
