import argparse
import globalSettings
import tokenizer.trainTokenizer as trainTokenizer
import tokenizer.tokenizer as tokenizerModule
import data.prepareDataset as prepareDataset
import model.train as trainModule
import generate as generateModule
import countMatrics
import utils.estimateVocabSize as estimateVocabSize
import utils.estimateEpochs as estimateEpochs
import utils.estimateLearningRate as estimateLearningRate


def fileArgumentsParse():
    parser = argparse.ArgumentParser(
        prog="Dumbotron Artificial Unintelligence",
        description="A small Transformer-based Large Language Model built from scratch using PyTorch."
    )

    parser.add_argument(
        "--tokenize",
        action="store_true",
        help="Train the tokenizer and tokenize the dataset",
    )
    parser.add_argument(
        "--prepareDataset", action="store_true", help="Prepare the dataset for training"
    )
    parser.add_argument(
        "--train", action="store_true", help="Train the transformer model"
    )
    parser.add_argument(
        "--generate", action="store_true", help="Generate text from a prompt"
    )
    parser.add_argument(
        "--countMatrics",
        action="store_true",
        help="Count the number of matrics in the model",
    )

    # Secondary Args
    parser.add_argument(
        "--prompt", type=str, default="", help="Prompt to generate from"
    )
    parser.add_argument(
        "--dataset", type=str, default="./data/dataset/dataset.txt", help="Select dataset location"
    )
    parser.add_argument(
        "--maxNewTokens", type=int, default=200, help="Max tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument("--topK", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--contextWindow", type=int,default=globalSettings.CONTEXT_WINDOW, help="Context window size")

    return parser.parse_args()


def main():
    args = fileArgumentsParse()

    VOCAB_SIZE = estimateVocabSize.estimateVocabSize(args.dataset)
    EPOCHS = estimateEpochs.estimateEpochs(args.dataset)
    LEARNING_RATE = estimateLearningRate.estimateLearningRate(args.dataset)

    if args.tokenize:
        trainTokenizer.trainTokenizer(
            args.dataset,
            globalSettings.TOKENIZER_PREFIX,
            VOCAB_SIZE,
        )
        tokenizerModule.tokenizer(args.dataset)

    if args.prepareDataset:
        prepareDataset.prepareDataset()

    if args.train:
        trainModule.train(
            vocabSize=VOCAB_SIZE,
            embedDim=globalSettings.EMBED_DIM,
            numHeads=globalSettings.NUM_HEADS,
            ffDim=globalSettings.FF_DIM,
            numLayers=globalSettings.NUM_LAYERS,
            encodingBase=globalSettings.ENCODING_BASE,
            contextWindow=globalSettings.CONTEXT_WINDOW,
            dropout=globalSettings.DROPOUT,
            batchSize=globalSettings.BATCH_SIZE,
            learningRate=LEARNING_RATE,
            epochs=EPOCHS,
            modelSavePath=globalSettings.MODEL_SAVE_PATH,
        )

    if args.generate:
        output = generateModule.generate(
            prompt=args.prompt,
            vocabSize=VOCAB_SIZE,
            embedDim=globalSettings.EMBED_DIM,
            numHeads=globalSettings.NUM_HEADS,
            ffDim=globalSettings.FF_DIM,
            numLayers=globalSettings.NUM_LAYERS,
            encodingBase=globalSettings.ENCODING_BASE,
            contextWindow=globalSettings.CONTEXT_WINDOW,
            modelSavePath=globalSettings.MODEL_SAVE_PATH,
            tokenizerPath=globalSettings.TOKENIZER_PREFIX + ".model",
            maxNewTokens=args.maxNewTokens,
            temperature=args.temperature,
            topK=args.topK,
        )
        print(output)
    
    if args.countMatrics:
        countMatrics.countMatrics(globalSettings.MODEL_SAVE_PATH, args.dataset)


if __name__ == "__main__":
    main()
