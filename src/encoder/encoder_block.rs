use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::ResidualConnection,
};

pub struct EncoderBlock {
    mha: MultiHeadAttnBlock,
    ff: FeedForwardBlock,
    rconns: [ResidualConnection; 2],
}

impl EncoderBlock {
    pub fn new(mha: MultiHeadAttnBlock, ff: FeedForwardBlock, dropout: f32) -> Result<Self> {
        let rconn1 = ResidualConnection::new(dropout)?;
        let rconn2 = ResidualConnection::new(dropout)?;
        Ok(EncoderBlock {
            mha,
            ff,
            rconns: [rconn1, rconn2],
        })
    }

    pub fn forward(&self, xs: &Tensor, src_mask: Option<Tensor>) -> Result<Tensor> {
        let x = self.rconns[0].forward(xs, src_mask, &self.mha)?;
        let x = self.rconns[1].forward(&x, None, &self.ff)?;
        Ok(x)
    }
}
