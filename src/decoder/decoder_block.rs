use std::rc;

use candle_core::{Result, Tensor};
use candle_nn::Dropout;

use crate::{
    feed_forward::feed_forward::FeedForwardBlock,
    multi_head_attn::multihead_block::MultiHeadAttnBlock,
    residual_layer::residual_conn::{ResidualConnection, SubLayers},
};

/// The `DecoderBlock` represents a single block in the decoder part of the Transformer architecture.
/// It consists of a multi-head self-attention mechanism, a multi-head cross-attention mechanism,
/// a feed-forward neural network, and a set of residual connections.
#[derive(Debug)]
pub struct DecoderBlock<'a> {
    /// The multi-head self-attention block for attending to the previous output elements.
    self_attn: MultiHeadAttnBlock<'a>,
    /// The multi-head cross-attention block for attending to the encoder output.
    cross_attn: MultiHeadAttnBlock<'a>,
    /// The feed-forward neural network block for further processing the attention outputs.
    ff: FeedForwardBlock,
    /// The residual connections applied after each sub-layer.
    rconns: Vec<ResidualConnection>,
}

impl<'a> DecoderBlock<'a> {
    /// Creates a new `DecoderBlock` instance with the given sub-components and dropout rate.
    ///
    /// # Arguments
    ///
    /// * `self_attn` - The multi-head self-attention block.
    /// * `cross_attn` - The multi-head cross-attention block.
    /// * `ff` - The feed-forward neural network block.
    /// * `dropout` - The dropout rate to apply in the residual connections.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the new `DecoderBlock` instance on success, or an error if
    /// the residual connections could not be created.
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
    /// Performs the forward pass through the decoder block.
    ///
    /// # Arguments
    ///
    /// * `xs` - The input tensor representing the previous decoder outputs.
    /// * `encoder_output` - The output tensor from the encoder part of the Transformer.
    /// * `src_mask` - A boolean indicating whether to apply a mask to the encoder output.
    /// * `tgt_mask` - A boolean indicating whether to apply a mask to the previous decoder outputs.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the output tensor after applying the decoder block operations,
    /// or an error if any of the sub-layer operations failed.
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
