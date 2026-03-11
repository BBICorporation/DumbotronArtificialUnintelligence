import sentencepiece as spm

# Train the tokenizer to produce
# a vocabulary of 148 tokens
# and a BPE model
spm.SentencePieceTrainer.train(
    input="data/dataset.txt",
    model_prefix="tokenize/tokenizer",
    vocab_size=148,
    model_type="bpe",
    character_coverage=1.0,
    bos_id=1,
    eos_id=2,
    pad_id=0,
    unk_id=3,
)

print("Tokenizer trained!")
