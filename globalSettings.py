# --- Tokenizer ---
TOKENIZER_PREFIX = "tokenizer/data/tokenizer"

# --- Model ---
EMBED_DIM = 256
NUM_HEADS = 8
FF_DIM = 1024
NUM_LAYERS = 4
ENCODING_BASE = 10000.0
CONTEXT_WINDOW = 128
DROPOUT = 0.1

# --- Training ---
BATCH_SIZE = 64
MODEL_SAVE_PATH = "model/saved/transformer.pt"
