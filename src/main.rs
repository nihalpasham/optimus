use candle_core::Device;
use tokenizers::tokenizer::{Result, Tokenizer};

use crate::embeddings::input_embeddings::InputEmbeddings;

mod embeddings;
mod layer_norm;
// mod testspace;
mod tokenizer;

const D_MODEL: usize = 512;

fn main() -> Result<()> {
    // load a pre-trained tokenizer
    let tokenizer = Tokenizer::from_file("./src/tokenizer/wordlevel-wiki.json")?;

    let encoding = tokenizer.encode(("Welcome to the library. ", "test this out"), true)?;
    println!("tok:  {:?}", encoding.get_tokens());
    // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
    println!("ids:  {:?}", encoding.get_ids());
    // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]

    let vocab_size = tokenizer.get_vocab_size(true);
    let token_ids = encoding.get_ids();

    let device = Device::new_metal(0)?;
    let input_embeds = InputEmbeddings::new(vocab_size, D_MODEL, &device)?;
    let embeddings = input_embeds.forward(&token_ids, &device)?;

    println!("vector embeddings: {}", embeddings);

    Ok(())
}
