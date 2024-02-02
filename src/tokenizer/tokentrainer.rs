use tokenizers::{
    models::{
        wordlevel::{WordLevel, WordLevelTrainer},
        TrainerWrapper,
    },
    normalizers::{utils::Sequence as NormalizerSequence, Lowercase, StripAccents, NFD},
    pre_tokenizers::{digits::Digits, sequence::Sequence, whitespace::Whitespace},
    processors::template::TemplateProcessing,
    tokenizer::{Result, Tokenizer},
    AddedToken,
};

#[allow(dead_code)]
pub fn wordlevel_tokenizer() -> Result<()> {
    // WordLevel is a not a sub-word tokenizer
    let mut tokenizer = Tokenizer::new(WordLevel::default());

    // normalize input with a specified sequence of normalizers
    let normalizer =
        NormalizerSequence::new(vec![NFD.into(), StripAccents.into(), Lowercase.into()]);
    tokenizer.with_normalizer(normalizer);

    // pre-tokenize input i.e. split our input text using a sequence of pre-tokenizers
    let pre_tokenizer = Sequence::new(vec![Whitespace.into(), Digits::new(true).into()]);
    tokenizer.with_pre_tokenizer(pre_tokenizer);

    // tokenizer's model trainer
    let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
        .vocab_size(30_522)
        .special_tokens(vec![
            AddedToken::from("[UNK]", true),
            AddedToken::from("[CLS]", true),
            AddedToken::from("[SEP]", true),
            AddedToken::from("[PAD]", true),
            AddedToken::from("[MASK]", true),
        ])
        .build()?
        .into();

    let files = vec![
        "./src/tokenizer/wikitext-103-raw/wiki.train.raw".into(),
        "./src/tokenizer/wikitext-103-raw/wiki.test.raw".into(),
        "./src/tokenizer/wikitext-103-raw/wiki.valid.raw".into(),
    ];
    // train tokenizer, given training, testing and validation datasets.
    // training the tokenizer means it will learn using the following rules:

    // - Start with all the characters present in the training corpus as tokens.
    // - Identify the most common pair of tokens and merge it into one token.
    // - Repeat until the vocabulary (e.g., the number of tokens) has reached the size we want.
    tokenizer.train_from_files(&mut trainer, files)?;
    // post processor
    tokenizer.with_post_processor(
        TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .unwrap()
            .try_pair("[CLS] $A [SEP] $B:1 [SEP]:1")
            .unwrap()
            .special_tokens(vec![("[CLS]", 1), ("[SEP]", 2)])
            .build()
            .unwrap(),
    );
    // save tokenizer
    tokenizer.save("./wordlevel-wiki.json", false)?;

    let encoding = tokenizer.encode(("Welcome to the library.", "test this out"), true)?;
    println!("tok: {:?}", encoding.get_tokens());
    // tok:  ["welcome", "to", "the", "library", ".", "test", "this", "out"]
    println!("ids:  {:?}", encoding.get_ids());
    // ids:  [5807, 11, 5, 1509, 7, 681, 48, 92]
    println!("sid:  {:?}", encoding.get_sequence_ids());
    // sid:  [Some(0), Some(0), Some(0), Some(0), Some(0), Some(1), Some(1), Some(1)]

    // given the position of a token in the sequence, returns the sequence id and a tuple containing the start and end
    // positions of chars in the sequence.
    println!("ttc:  {:?}", encoding.token_to_chars(2));
    // ttc:  Some((0, (11, 14)))

    // given the position of a char in the sequence, returns the position of the token in the sequence
    println!("ctt:  {:?}", encoding.char_to_token(11, 0));
    // ctt:  Some(2)

    println!("wid:  {:?}", encoding.get_word_ids());
    // wid:  [Some(0), Some(1), Some(2), Some(3), Some(4), Some(0), Some(1), Some(2)]
    println!("idt:  {:?}", tokenizer.decode(&[4058], true));
    // idt:  Ok("hindu")
    println!("ids:  {:?}", tokenizer.decode(encoding.get_ids(), true));

    Ok(())
}
