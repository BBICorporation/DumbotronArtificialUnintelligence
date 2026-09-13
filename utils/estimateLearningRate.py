def estimateLearningRate(DATASET_PATH):
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    numWords = len(text.split())

    if numWords < 50_000:
        return 1e-4
    elif numWords < 200_000:
        return 2e-4
    elif numWords < 1_000_000:
        return 3e-4
    elif numWords < 5_000_000:
        return 3e-4
    else:
        return 2e-4