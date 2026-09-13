import torch
import numpy as np
import sentencepiece as spm


def countMatrics(MODEL_SAVE_PATH, DATASET_PATH):
    # Load model
    state_dict = torch.load(MODEL_SAVE_PATH, map_location="cpu")

    # Count parameters
    total_params = sum(p.numel() for p in state_dict.values())

    # Load token IDs
    token_ids = np.load("data/input.npy")

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load("tokenizer/data/tokenizer.model")

    # Load dataset
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Statistics
    total_tokens = len(token_ids)
    vocab_size = sp.GetPieceSize()
    max_token_id = token_ids.max()

    # Print results
    print("=" * 40)
    print("Model Statistics")
    print("=" * 40)

    print(f"Total parameters : {total_params:,}")
    print(f"Parameters       : {total_params / 1e6:.2f} M")

    print()
    print("=" * 40)
    print("Token Statistics")
    print("=" * 40)

    print(f"Total tokens     : {total_tokens:,}")
    print(f"Vocab size       : {vocab_size:,}")
    print(f"Max token ID     : {max_token_id:,}")

    print()
    print("=" * 40)
    print("Dataset Statistics")
    print("=" * 40)

    print(f"Total sentences  : {len(lines):,}")

    print("=" * 40)