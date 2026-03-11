# DumbotronArtificialUnintelligence

A deliberately tiny LLM (≈5K – 3M parameters approx) built with **TensorFlow** and runnable locally via **Ollama** after conversion.

---

> [!NOTE]
> # Contribution Guidelines
> 
> To keep the codebase consistent, follow these naming conventions:
> 
> **Constants**: Use **SCREAMING_SNAKE_CASE** = `MAX_CONTEXT_LENGTH`
> 
> **Functions**: Use **PascalCase** = `BuildTokenizer()`
> 
> **Variables & File names**: Use **camelCase** = `tokenSequence`

---

> [!IMPORTANT]
> # Workflow
> 
> The project follows a simple pipeline:
> 
> ```
> Environment Setup
>       ↓
> Data Collection
>       ↓
> Tokenizer Training
>       ↓
> Tokenize Dataset
>       ↓
> Prepare Training Sequences
>       ↓
> Build Model
>       ↓
> Train Model
>       ↓
> Save Model
>       ↓
> Convert to GGUF
>       ↓
> Run with Ollama
> ```
> 
> ---
> 
> ## 1. Environment Setup
> 
> Install the required tools and libraries.
> 
> Purpose:
> 
> * Create a Python ML environment
> * Install TensorFlow and supporting libraries for training
> 
> Typical setup includes:
> 
> * Python 3.9
> * TensorFlow
> * SentencePiece
> * NumPy
> * tqdm
> 
> This step prepares the system for dataset processing and model training.
> 
> ---
> 
> ## 2. Data Collection
> 
> Collect text data that the model will learn from.
> 
> Possible sources:
> 
> * small story datasets
> * Wikipedia text
> * books
> * custom `.txt` files
> * code repositories
> 
> For a small model, a few MB to a few hundred MB of text is sufficient.
> 
> The dataset is typically stored as plain text files.
> 
> ---
> 
> ## 3. Tokenizer Training
> 
> Neural networks cannot process raw text directly, so a tokenizer converts text into numeric tokens.
> 
> Tokenizer training scans the dataset and builds a vocabulary of common words or subwords.
> 
> Output of this step:
> 
> ```
> tokenizer.model
> tokenizer.vocab
> ```
> 
> This tokenizer will be used during both training and inference.
> 
> ---
> 
> ## 4. Tokenizing the Dataset
> 
> After the tokenizer is created, the entire dataset is converted into token IDs.
> 
> Example concept:
> 
> ```
> text → tokens
> ```
> 
> The dataset becomes a long sequence of integers representing tokens.
> 
> These tokens are what the model actually trains on.
> 
> ---
> 
> ## 5. Preparing Training Sequences
> 
> LLMs train on **context windows**.
> 
> The token sequence is split into smaller chunks of fixed length (context size).
> 
> Each chunk is used for **next-token prediction**.
> 
> Input sequence:
> 
> ```
> [token1, token2, token3]
> ```
> 
> Target sequence:
> 
> ```
> [token2, token3, token4]
> ```
> 
> The model learns to predict the next token in a sequence.
> 
> ---
> 
> ## 6. Building the Model
> 
> The model architecture is a small **Transformer decoder**.
> 
> Main components:
> 
> * Token embedding layer
> * Positional encoding
> * Transformer blocks
> * Feed-forward network
> * Output projection layer
> 
> The size of the model depends on:
> 
> * embedding dimension
> * number of layers
> * number of attention heads
> * vocabulary size
> 
> ---
> 
> ## 7. Training the Model
> 
> During training the following happens repeatedly:
> 
> 1. A batch of token sequences is loaded.
> 2. The sequences pass through the neural network.
> 3. The model predicts next-token probabilities.
> 4. Predictions are compared with actual tokens.
> 5. A loss value is calculated.
> 6. Model weights are updated using backpropagation.
> 
> This process gradually improves the model's predictions.
> 
> ---
> 
> ## 8. Saving the Model
> 
> After training completes, the following artifacts are saved:
> 
> * trained model weights
> * tokenizer
> * model configuration
> 
> These files represent the trained language model.
> 
> ---
> 
> ## 9. Model Conversion
> 
> TensorFlow models cannot run directly in Ollama.
> 
> The trained model must be converted into **GGUF format**, which is used by llama.cpp and Ollama.
> 
> Conversion typically involves:
> 
> ```
> TensorFlow model
>       ↓
> weight conversion
>       ↓
> GGUF model file
> ```
> 
> The GGUF file contains the model weights and architecture information.
> 
> ---
> 
> ## 10. Running with Ollama
> 
> Once converted, the model can be loaded into Ollama using a **Modelfile**.
> 
> The Modelfile specifies:
> 
> * the model file location
> * runtime parameters (temperature, context size)
> 
> After creating the model, it can be run locally through Ollama to generate text.
> 
