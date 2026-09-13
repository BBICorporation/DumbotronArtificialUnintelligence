import sentencepiece as spm
import numpy as np

def tokenizer(DATASET_PATH):
    # Load the trained tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/data/tokenizer.model")
    
    # Encode the text
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    
    tokenIds = sp.encode(text)
    
    # Displaying some dokens
    print("First 50 tokens:", tokenIds[:50])
    print("Total token: ", len(tokenIds))
    print("Vocab size:", sp.vocab_size())
    
    # Saving the tokenids
    np.save("data/tokenIds.npy", tokenIds)
    
    print("Saved tokenIds.npy")
    
