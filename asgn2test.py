import collections
import re
import math


def preprocess_text(filename):
    """ Reads the file and tokenizes it, returning a list of tokens. """
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read().lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def build_vocab(tokens, min_count=3):
    """ Build vocabulary where any token that occurs fewer than `min_count` times
        is replaced with <UNK>. """
    # Count token frequencies in the training data
    token_counts = collections.Counter(tokens)

    vocab = {}
    for token, count in token_counts.items():
        if count >= min_count:
            vocab[token] = count
        else:
            vocab['<UNK>'] = vocab.get('<UNK>', 0) + count

    # Add <STOP> token explicitly to the vocabulary
    vocab['<STOP>'] = 1
    print(f"Vocabulary has {len(vocab)} unique tokens, including <UNK> and <STOP>")

    return vocab


def replace_oov_tokens(tokens, vocab):
    """ Replace tokens not in vocab with <UNK> """
    replaced_tokens = 0
    updated_tokens = []
    for token in tokens:
        if token in vocab:
            updated_tokens.append(token)
        else:
            updated_tokens.append('<UNK>')
            replaced_tokens += 1
    print(f"Replaced {replaced_tokens} OOV tokens with <UNK>")
    return updated_tokens


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


def calculate_perplexity(model, tokens, ngram_type="unigram", bigram_model=None, trigram_model=None):
    """ Calculate perplexity of a given model on a test set. """
    total_log_likelihood = 0
    word_count = 0

    # Handle Unigram, Bigram, and Trigram models differently
    if ngram_type == "unigram":
        # For unigram model, iterate over each token in the test dataset
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

    train_tokens = preprocess_text(train_file)
    test_tokens = preprocess_text(test_file)

    vocab = build_vocab(train_tokens, min_count=3)

    train_tokens = replace_oov_tokens(train_tokens, vocab)
    test_tokens = replace_oov_tokens(test_tokens, vocab)

    print(f"Total unique tokens in the vocabulary: {len(vocab)}")

    print("Building unigram, bigram, and trigram models...")
    unigram_model, total_tokens = build_unigram_model(train_tokens)
    bigram_model, unigram_counts = build_bigram_model(train_tokens)
    trigram_model, bigram_counts = build_trigram_model(train_tokens)

    print("Calculating perplexities...")
    unigram_perplexity = calculate_perplexity(unigram_model, test_tokens, ngram_type="unigram")
    bigram_perplexity = calculate_perplexity(None, test_tokens, ngram_type="bigram", bigram_model=bigram_model)
    trigram_perplexity = calculate_perplexity(None, test_tokens, ngram_type="trigram", trigram_model=trigram_model)

    print(f"Unigram Perplexity: {unigram_perplexity}")
    print(f"Bigram Perplexity: {bigram_perplexity}")
    print(f"Trigram Perplexity: {trigram_perplexity}")

if __name__ == "__main__":
    main()