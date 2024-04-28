use std::rc;

use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::{ResidualConnection, SubLayers},
};

#[derive(Debug)]
pub struct DecoderBlock<'a> {
    self_attn: MultiHeadAttnBlock<'a>,
    cross_attn: MultiHeadAttnBlock<'a>,
    ff: FeedForwardBlock,
    rconns: Vec<ResidualConnection>,
}

impl<'a> DecoderBlock<'a> {
    pub fn new(
        self_attn: MultiHeadAttnBlock<'a>,
        cross_attn: MultiHeadAttnBlock<'a>,
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
        src_mask: bool,
        tgt_mask: bool,
    ) -> Result<Tensor> {
        let x = self.rconns[0].forward(xs, None, src_mask, SubLayers::Mha(&self.self_attn))?;
        let x = self.rconns[1].forward(
            &x,
            Some(encoder_output),
            tgt_mask,
            SubLayers::Mha(&self.cross_attn),
        )?;
        let x = self.rconns[2].forward(&x, None, false, SubLayers::Ff(&self.ff))?;
        Ok(x)
    }
}
