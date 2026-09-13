def estimateEpochs(DATASET_PATH):
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        text = file.read()

    numWords = len(text.split())

    if numWords < 50_000:
        return 20
    elif numWords < 200_000:
        return 15
    elif numWords < 1_000_000:
        return 10
    elif numWords < 5_000_000:
        return 5
    else:
        return 3