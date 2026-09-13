import sentencepiece as spm

# Train the tokenizer
def trainTokenizer(DATASET_PATH, TOKENIZER_PREFIX, VOCAB_SIZE):
    spm.SentencePieceTrainer.train(
        input=DATASET_PATH,
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
    )

    print("Tokenizer trained!")
