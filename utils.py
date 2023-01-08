import os
from csv import reader
import nltk

import file_manager
from commons import (Corpus, DataPath,
                     SENTIMENT_CORPORA, GERMAN_SPACY_MODEL, GERMAN_STOPWORDS)


def generate_corpora():
    positive_reviews = []
    neutral_reviews = []
    negative_reviews = []


    # skip first line i.e. read header first and then iterate over each row od csv as a list
    with open(str(DataPath.TRAIN.value), 'r', encoding='utf-8') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header is not None:
            # Iterate over each row after the header in the csv
            for row in csv_reader:
                # row variable is a list that represents a row in csv
                star_rating = row[1]
                if star_rating == "0":
                    positive_reviews.append(row[0])
                elif star_rating == "1":
                    neutral_reviews.append(row[0])
                elif star_rating == "2":
                    negative_reviews.append(row[0])

    with open(file_manager.get_filepath(f"corpora_nlp22z/{Corpus.POSITIVE.value}"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(positive_reviews))

    with open(file_manager.get_filepath(f"corpora_nlp22z/{Corpus.NEUTRAL.value}"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(neutral_reviews))

    with open(file_manager.get_filepath(f"corpora_nlp22z/{Corpus.NEGATIVE.value}"), 'w', encoding='utf-8') as f:
        f.write('\n'.join(negative_reviews))


def create_top10_bigrams():
    positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in SENTIMENT_CORPORA.words(Corpus.POSITIVE.value)
        if w.isalpha() and w.lower() not in GERMAN_STOPWORDS
    ])
    neutral_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in SENTIMENT_CORPORA.words(Corpus.NEUTRAL.value)
        if w.isalpha() and w.lower() not in GERMAN_STOPWORDS
    ])
    negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
        w for w in SENTIMENT_CORPORA.words(Corpus.NEGATIVE.value)
        if w.isalpha() and w.lower() not in GERMAN_STOPWORDS
    ])

    bigram_pos = [w[0] for w in positive_bigram_finder.ngram_fd.most_common(100)]
    bigram_neu = [w[0] for w in neutral_bigram_finder.ngram_fd.most_common(100)]
    bigram_neg = [w[0] for w in negative_bigram_finder.ngram_fd.most_common(100)]
    bigram_pos = [(left.lower(), right.lower()) for left, right in bigram_pos]
    bigram_neu = [(left.lower(), right.lower()) for left, right in bigram_neu]
    bigram_neg = [(left.lower(), right.lower()) for left, right in bigram_neg]

    bigram_pos_dict = dict(w for w in positive_bigram_finder.ngram_fd.most_common(100))
    bigram_neu_dict = dict(w for w in neutral_bigram_finder.ngram_fd.most_common(100))
    bigram_neg_dict = dict(w for w in negative_bigram_finder.ngram_fd.most_common(100))

    common_pos_neu = [word for word in (set(bigram_pos) & set(bigram_neu))]
    common_pos_neg = [word for word in (set(bigram_pos) & set(bigram_neg))]
    common_neu_neg = [word for word in (set(bigram_neu) & set(bigram_neg))]
    common_bigrams = set(common_pos_neu + common_pos_neg + common_neu_neg)

    for bigram in common_bigrams:
        try:
            pos_freq = bigram_pos_dict[bigram]
        except:
            pos_freq = 0
        try:
            neu_freq = bigram_neu_dict[bigram]
        except:
            neu_freq = 0
        try:
            neg_freq = bigram_neg_dict[bigram]
        except:
            neg_freq = 0
        max_freq = max(pos_freq, neu_freq, neg_freq)
        if pos_freq == max_freq:
            bigram_neu.remove(bigram) if bigram in bigram_neu else None
            bigram_neg.remove(bigram) if bigram in bigram_neg else None
        elif neu_freq == max_freq:
            bigram_pos.remove(bigram) if bigram in bigram_pos else None
            bigram_neg.remove(bigram) if bigram in bigram_neg else None
        elif neg_freq == max_freq:
            bigram_pos.remove(bigram) if bigram in bigram_pos else None
            bigram_neu.remove(bigram) if bigram in bigram_neu else None

    return bigram_pos[:10], bigram_neu[:10], bigram_neg[:10]


def save_unwanted_words(unwanted_words):
    unwanted_words_cached = load_unwanted_words()
    words_to_save = []
    for word in unwanted_words:
        # if not cached already, then save
        if word not in unwanted_words_cached:
            words_to_save.append(word)
    with open(str(DataPath.CACHE.value), "a+", encoding="utf-8") as f:
        for word in words_to_save:
            f.write(f"{word}\n")
    print(f"Saved {len(words_to_save)} new unwanted words!")

def load_unwanted_words():
    unwanted_words = []
    if not os.path.isfile(str(DataPath.CACHE.value)):
        return []
    with open(str(DataPath.CACHE.value), "r", encoding="utf-8") as f:
        for word in f.readlines():
            unwanted_words.append(word.strip())
    return unwanted_words
def skip_unwanted(word, unwanted_words_log):
    # if already cached as unwanted
    if word in unwanted_words_log:
        return False

    # we don't want non-alphanumeric or stopwords
    if not word.isalpha() or word.lower() in GERMAN_STOPWORDS:
        unwanted_words_log.append(word)
        return False

    # we dont want nouns
    for t in GERMAN_SPACY_MODEL(word):
        if t.pos_ in ("NOUN", "PROPN"):
            unwanted_words_log.append(word)
            return False
    return True

def generate_unwanted_words_cache():
    unwanted_words_log = []
    print("Generating unwanted words cache...")
    print("Looking for unwanted words (positive corpus)...")
    for word in SENTIMENT_CORPORA.words(Corpus.POSITIVE.value):
        skip_unwanted(word, unwanted_words_log=unwanted_words_log)
    save_unwanted_words(unwanted_words_log)

    print("Looking for unwanted words (neutral corpus)...")
    for word in SENTIMENT_CORPORA.words(Corpus.NEUTRAL.value):
        skip_unwanted(word, unwanted_words_log=unwanted_words_log)
    save_unwanted_words(unwanted_words_log)

    print("Looking for unwanted words (negative corpus)...")
    for word in SENTIMENT_CORPORA.words(Corpus.NEGATIVE.value):
        skip_unwanted(word, unwanted_words_log=unwanted_words_log)
    save_unwanted_words(unwanted_words_log)
