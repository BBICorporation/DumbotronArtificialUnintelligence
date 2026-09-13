import os
import sentencepiece as spm


def estimateVocabSize(DATASET_PATH):
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    num_chars = len(text)
    num_words = len(text.split())

    # Simple heuristic
    VOCAB_SIZE = int(num_words ** 0.5 * 10)

    # Keep it within reasonable bounds
    VOCAB_SIZE = max(500, min(VOCAB_SIZE, 16000))

    return VOCAB_SIZE