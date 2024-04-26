use std::rc;

use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::{ResidualConnection, SubLayers},
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
        let mut rconns = Vec::with_capacity(3);
        for conn in 0..3 {
            rconns.push(ResidualConnection::new(dropout)?);
        }
        Ok(DecoderBlock {
            self_attn,
            cross_attn,
            ff,
            rconns,
        })
    }
 
    pub fn forward(
        &self,
        xs: &Tensor,
        encoder_output: &Tensor,
        src_mask: Option<Tensor>,
        tgt_mask: Option<Tensor>,
    ) -> Result<Tensor> {
        let x = self.rconns[0].forward(xs, None, src_mask, SubLayers::Mha(&self.self_attn))?;
        let x = self.rconns[1].forward(
            &x,
            Some(encoder_output),
            tgt_mask,
            SubLayers::Mha(&self.cross_attn),
        )?;
        let x = self.rconns[2].forward(&x, None, None, SubLayers::Ff(&self.ff))?;
        Ok(x)
    }
}
