# Dumbotron Artificial Unintelligence

A small **Transformer-based Large Language Model built from scratch using PyTorch**.

The project focuses on understanding how LLMs work internally by implementing the core components without using a pre-built Transformer model.

## Tech Stack

- Python
- PyTorch
- CUDA
- SentencePiece
- NumPy

## Project Structure

```
DumbotronArtificialUnintelligence/
├── data
│   ├── datasets
│   │   ├── dataset_44001.txt
│   │   └── dataset_50.txt
│   └── prepareDataset.py
├── model
│   ├── attention.py
│   ├── feedForward.py
│   ├── positionEncoding.py
│   ├── tokenEmbeding.py
│   ├── train.py
│   ├── transformerBlock.py
│   └── transformer.py
├── tokenizer
│   ├── tokenizer.py
│   └── trainTokenizer.py
├── utils
│   ├── estimateEpochs.py
│   ├── estimateLearningRate.py
│   └── estimateVocabSize.py
├── generate.py
├── globalSettings.py
├── main.py
├── requirments.txt
├── countMatrics.py
└── README.md
```

## Pipeline

```
Dataset
   ↓
Tokenizer
   ↓
Token IDs
   ↓
Transformer
   ↓
Training
   ↓
Text Generation
```

## Installation

```
pip install -r requirments.txt
```

## Usage

### Train tokenizer

```
python main.py --tokenize
```

### Prepare dataset

```
python main.py --prepareDataset
```

### Train model

```
python main.py --train
```

### Generate text

```
python main.py --generate --prompt "Hello"
```

Optional generation settings:

```
python main.py --generate \
    --prompt "Hello" \
    --maxNewTokens 200 \
    --temperature 1.0 \
    --topK 50
```

### View model statistics

```
python main.py --countMatrics
```

This shows information such as parameter count, token count, vocabulary size, maximum token ID, and number of sentences.

## Architecture

The model is built from basic Transformer components:

```
Token Embedding
      ↓
Positional Encoding
      ↓
Transformer Blocks
      ├── Self Attention
      └── Feed Forward
      ↓
Output
```

## Project Goal

The goal of **Dumbotron Artificial Unintelligence** is to learn and experiment with the internals of Transformer-based language models by building one from scratch.

> Educational and experimental project — expect both intelligent output and complete nonsense.

## License

See [`COPYING`](COPYING).