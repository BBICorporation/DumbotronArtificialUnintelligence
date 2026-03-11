import sentencepiece as spm
import numpy as np

# Load the trained tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenize/tokenizer.model")

# Encode the text
with open("data/dataset.txt", "r", encoding="utf-8") as f:
    text = f.read()

token_ids = sp.encode(text)

# Displaying some dokens
print("First 50 tokens:", token_ids[:50])
print("Vocab size:", sp.vocab_size())

# Saving the tokenids
np.save("data/tokenIds.npy", token_ids)

print("Saved token_ids.npy")
