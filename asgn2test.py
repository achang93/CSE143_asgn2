import collections
import re
import math

def preprocess(data_file, min_freq=3):
    #Preprocesses the data from a .tokens file to handle the <UNK> token and adds <START> and <STOP> tokens.
    with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
        sentences = [line.strip().split() for line in f]

    tokens = [token for sentence in sentences for token in sentence]
    token_counts = collections.Counter( tokens)

    print(f"Number of unique tokens in {data_file}: {len(token_counts)}")

    for sentence in sentences:
        for i, token in enumerate(sentence):
            if token_counts[token] < min_freq:
                sentence[i] = "<UNK>"

    sentences = [["<START>"] + sentence + ["<STOP>"] for sentence in sentences]

    return sentences, token_counts


def build_unigram_model(tokens):
    """ Build a unigram model based on the tokenized training data. """
    unigram_counts = collections.Counter(tokens)
    total_tokens = sum(unigram_counts.values())
    unigram_model = {word: count / total_tokens for word, count in unigram_counts.items()}
    return unigram_model, total_tokens


def build_bigram_model(tokens):
    """ Build a bigram model based on the tokenized training data. """
    bigram_counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
    unigram_counts = collections.Counter(tokens)
    bigram_model = {
        (w1, w2): count / unigram_counts[w1] for (w1, w2), count in bigram_counts.items()
    }
    return bigram_model, unigram_counts


def build_trigram_model(tokens):
    """ Build a trigram model based on the tokenized training data. """
    trigram_counts = collections.Counter(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    bigram_counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
    trigram_model = {
        (w1, w2, w3): count / bigram_counts[(w1, w2)] for (w1, w2, w3), count in trigram_counts.items()
    }
    return trigram_model, bigram_counts


# Implementing additive smoothing for unigram, bigram, and trigram models
def build_unigram_smooth_model(tokens, alpha=1):
    """ Build a unigram model with additive smoothing. """
    unigram_counts = collections.Counter(tokens)
    total_tokens = sum(unigram_counts.values()) + alpha * len(set(tokens))
    unigram_model = {word: (count + alpha) / total_tokens for word, count in unigram_counts.items()}
    return unigram_model, total_tokens


def build_bigram_smooth_model(tokens, alpha=1):
    """ Build a bigram model with additive smoothing. """
    bigram_counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
    unigram_counts = collections.Counter(tokens)
    vocabulary = len(set(tokens))
    bigram_model = {
        (w1, w2): (count + alpha) / (unigram_counts[w1] + alpha * vocabulary) for (w1, w2), count in bigram_counts.items()
    }
    return bigram_model, unigram_counts


def build_trigram_smooth_model(tokens, alpha=1):
    """ Build a trigram model with additive smoothing. """
    trigram_counts = collections.Counter(zip(tokens[:-2], tokens[1:-1], tokens[2:]))
    bigram_counts = collections.Counter(zip(tokens[:-1], tokens[1:]))
    vocabulary = len(set(tokens))
    trigram_model = {
        (w1, w2, w3): (count + alpha) / (bigram_counts[(w1, w2)] + alpha * vocabulary) for (w1, w2, w3), count in trigram_counts.items()
    }
    return trigram_model, bigram_counts

def build_smooth_interpol(tokens, l1 = 0.1, l2 = 0.3, l3 = 0.6):
    """ Build a model with interpolation. """
    unigram_model, total_tokens = build_unigram_model(tokens)
    bigram_model, unigram_counts = build_bigram_model(tokens)
    trigram_model, bigram_counts = build_trigram_model(tokens)
    interpol_model = {}
    for i in range(len(tokens)):
        unigramProb = unigram_model.get(tokens[i], 0)
        if i <= 0:
            bigramProb = 0
        else:
            bigramProb = bigram_model.get((tokens[i-1], tokens[i]), 0)
        if i <= 1:
            trigramProb = 0
        else:
            trigramProb = trigram_model.get((tokens[i-2], tokens[i-1], tokens[i]), 0)
        total = l1 * unigramProb + l2 * bigramProb + l3 * trigramProb
        if total == 0:
            continue
        interpol_model[(tokens[i-2], tokens[i-1], tokens[i])] = total
    return interpol_model

def calculate_perplexity(model, tokens, ngram_type="unigram", bigram_model=None, trigram_model=None):
    """ Calculate perplexity of a given model on a test set. """
    total_log_likelihood = 0
    word_count = 0

    if ngram_type == "unigram":
        for token in tokens:
            prob = model.get(token, 0)  # If token is <UNK>, it will be 0 or use some smoothing
            if prob > 0:
                total_log_likelihood += math.log(prob)
            word_count += 1

    elif ngram_type == "bigram":
        # For bigram model, iterate over each token from the second token onward
        for i in range(1, len(tokens)):
            prev_token = tokens[i - 1]
            current_token = tokens[i]
            prob = bigram_model.get((prev_token, current_token), 0)
            if prob > 0:
                total_log_likelihood += math.log(prob)
            word_count += 1

    elif ngram_type == "trigram":
        # For trigram model, iterate over each token from the third token onward
        for i in range(2, len(tokens)):
            prev_prev_token = tokens[i - 2]
            prev_token = tokens[i - 1]
            current_token = tokens[i]
            prob = trigram_model.get((prev_prev_token, prev_token, current_token), 0)
            if prob > 0:
                total_log_likelihood += math.log(prob)
            word_count += 1

    # Calculate perplexity
    perplexity = math.exp(-total_log_likelihood / word_count)
    return perplexity


def main():
    train_file = '1b_benchmark.train.tokens'
    test_file = '1b_benchmark.test.tokens'
    dev_file = '1b_benchmark.dev.tokens'

    # Use the new preprocess function for both training and test data
    train_sentences, train_token_counts = preprocess(train_file, min_freq=3)
    test_sentences, test_token_counts = preprocess(test_file, min_freq=3)
    dev_sentences, dev_token_counts = preprocess(dev_file, min_freq=3)

    # Build vocabulary using the token counts (from preprocessing)
    vocab = {token for token, count in train_token_counts.items() if count >= 3}
    
    # Add <UNK> to the vocabulary for rare words
    vocab.add("<UNK>")
    vocab.add("<STOP>")  # Explicitly add <STOP> token

    # Flatten the sentences again for processing
    train_tokens = [token for sentence in train_sentences for token in sentence]
    test_tokens = [token for sentence in test_sentences for token in sentence]
    dev_tokens = [token for sentence in dev_sentences for token in sentence]

    # Create unigram, bigram, and trigram models
    unigram_model, total_tokens = build_unigram_model(train_tokens)
    bigram_model, unigram_counts = build_bigram_model(train_tokens)
    trigram_model, bigram_counts = build_trigram_model(train_tokens)

    print(f"Total unique tokens in the training vocabulary: {len(vocab)}")

    print("Calculating perplexities...")
    unigram_perplexity = calculate_perplexity(unigram_model, test_tokens, ngram_type="unigram")
    bigram_perplexity = calculate_perplexity(bigram_model, test_tokens, ngram_type="bigram", bigram_model=bigram_model)
    trigram_perplexity = calculate_perplexity(trigram_model, test_tokens, ngram_type="trigram", trigram_model=trigram_model)

    print(f"Unigram Perplexity: {unigram_perplexity}")
    print(f"Bigram Perplexity: {bigram_perplexity}")
    print(f"Trigram Perplexity: {trigram_perplexity}")

    # Building unigram, bigram, and trigram models with additive smoothing
    unigram_smooth_model, total_tokens = build_unigram_smooth_model(train_tokens, 4)
    bigram_smooth_model, unigram_counts = build_bigram_smooth_model(train_tokens, 4)
    trigram_smooth_model, bigram_counts = build_trigram_smooth_model(train_tokens, 4)

    print("Calculate perplexities with additive smoothing")
    unigram_smooth_perplexity = calculate_perplexity(unigram_smooth_model, test_tokens, ngram_type="unigram")
    bigram_smooth_perplexity = calculate_perplexity(bigram_smooth_model, test_tokens, ngram_type="bigram", bigram_model=bigram_smooth_model)
    trigram_smooth_perplexity = calculate_perplexity(trigram_smooth_model, test_tokens, ngram_type="trigram", trigram_model=trigram_smooth_model)

    print(f"Unigram Additive Smoothing Perplexity: {unigram_smooth_perplexity}")
    print(f"Bigram Additive Smoothing Perplexity: {bigram_smooth_perplexity}")
    print(f"Trigram Additive Smoothing Perplexity: {trigram_smooth_perplexity}")
    interpol_model = build_smooth_interpol(train_tokens)
    interpol_perplexity = calculate_perplexity(interpol_model, test_tokens, ngram_type = "trigram", trigram_model = interpol_model)
    print(f"Interpolation Perplexity: {interpol_perplexity}")

if __name__ == "__main__":
    main()
