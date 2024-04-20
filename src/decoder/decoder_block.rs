use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::ResidualConnection,
};

#[derive(Debug)]
pub struct DecoderBlock {
    self_attn: MultiHeadAttnBlock,
    cross_attn: MultiHeadAttnBlock,
    ff: FeedForwardBlock,
    rconns: Vec<ResidualConnection>,
}

impl DecoderBlock {
    pub fn new(
        self_attn: MultiHeadAttnBlock,
        cross_attn: MultiHeadAttnBlock,
        ff: FeedForwardBlock,
        dropout: f32,
    ) -> Result<Self> {
        Ok(DecoderBlock {
            self_attn,
            cross_attn,
            ff,
            rconns: Vec::with_capacity(3),
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<Tensor>,
        tgt_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let x = self.rconns[0].forward(xs,  src_mask, &self.self_attn)?;
        let x = self.rconns[1].forward(xs, tgt_mask, &self.self_attn)?;
        let x = self.rconns[1].forward(&x, None, &self.ff)?;
        Ok(x)
    }
}
