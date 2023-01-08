import os.path
from csv import reader

import matplotlib.pyplot as plt
import nltk
import numpy
from nltk.corpus import PlaintextCorpusReader
import enum
import spacy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

from bigrams import get_top10_bigrams

BALANCED = True
GENERATE_UNWANTED_WORDS_CACHE = False
INCLUDE_BIGRAMS = True


class CorpusType(enum.Enum):

    POSITIVE_CORPUS = "positive_reviews_balanced.txt" if BALANCED else "positive_reviews.txt"
    NEUTRAL_CORPUS = "neutral_reviews_balanced.txt" if BALANCED else "neutral_reviews.txt"
    NEGATIVE_CORPUS = "negative_reviews_balanced.txt" if BALANCED else "negative_reviews.txt"


TRAIN_PATH = "filtered_data/train_balanced.csv" if BALANCED else "filtered_data/train.csv"
TEST_PATH = "filtered_data/test_balanced.csv" if BALANCED else "filtered_data/test.csv"
CACHE_PATH = "cache/pos_cache_balanced.txt" if BALANCED else "cache/pos_cache.txt"

sentiment_corpora = PlaintextCorpusReader('corpora_nlp22z', '.*')
nlp = spacy.load('de_core_news_sm')
stopwords = nltk.corpus.stopwords.words("german")


def save_unwanted_words(unwanted_words):
    unwanted_words_cached = load_unwanted_words()
    words_to_save = []
    for word in unwanted_words:
        # if not cached already, then save
        if word not in unwanted_words_cached:
            words_to_save.append(word)
    with open(CACHE_PATH, "a+", encoding="utf-8") as f:
        for word in words_to_save:
            f.write(f"{word}\n")
    print(f"Saved {len(words_to_save)} new unwanted words!")


def load_unwanted_words():
    unwanted_words = []
    if not os.path.isfile(CACHE_PATH):
        return []
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        for word in f.readlines():
            unwanted_words.append(word.strip())
    return unwanted_words

unwanted_words_log = load_unwanted_words()


def skip_unwanted(word):
    # if already cached as unwanted
    if word in unwanted_words_log:
        return False

    # we don't want non-alphanumeric or stopwords
    if not word.isalpha() or word.lower() in stopwords:
        unwanted_words_log.append(word)
        return False

    # we dont want nouns
    for t in nlp(word):
        if t.pos_ in ("NOUN", "PROPN"):
            unwanted_words_log.append(word)
            return False
    return True


if GENERATE_UNWANTED_WORDS_CACHE:
    print("Generating unwanted words cache...")
    print("Looking for unwanted words (positive corpus)...")
    positive_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)

    print("Looking for unwanted words (neutral corpus)...")
    neutral_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)

    print("Looking for unwanted words (negative corpus)...")
    negative_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)
    exit()
else:
    print("Filtering unwanted words...")
    positive_words = sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value)
    neutral_words = sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value)
    negative_words = sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value)
    unwanted_words_log = set(unwanted_words_log)

    unwanted_positive_words = unwanted_words_log & set(positive_words)
    unwanted_neutral_words = unwanted_words_log & set(neutral_words)
    unwanted_negative_words = unwanted_words_log & set(negative_words)

    positive_words = filter(lambda x: x not in unwanted_positive_words, positive_words)
    neutral_words = filter(lambda x: x not in unwanted_neutral_words, neutral_words)
    negative_words = filter(lambda x: x not in unwanted_negative_words, negative_words)



positive_fd = nltk.FreqDist([w.lower() for w in positive_words])
neutral_fd = nltk.FreqDist([w.lower() for w in neutral_words])
negative_fd = nltk.FreqDist([w.lower() for w in negative_words])

# remove duplicates but leaving the most frequent in given set
print("Removing duplicates from dataset...")
common_pos_neu = [word for word in (set(positive_fd) & set(neutral_fd))]
common_pos_neg = [word for word in (set(positive_fd) & set(negative_fd))]
common_neu_neg = [word for word in (set(neutral_fd) & set(negative_fd))]
common_words = set(common_pos_neu + common_pos_neg + common_neu_neg)
for word in common_words:
    pos_freq, neu_freq, neg_freq = positive_fd.freq(word), neutral_fd.freq(word), negative_fd.freq(word)
    max_freq = max(pos_freq, neu_freq, neg_freq)
    if pos_freq == max_freq:
        neutral_fd.pop(word) if word in neutral_fd else None
        negative_fd.pop(word) if word in negative_fd else None
    elif neu_freq == max_freq:
        positive_fd.pop(word) if word in positive_fd else None
        negative_fd.pop(word) if word in negative_fd else None
    elif neg_freq == max_freq:
        positive_fd.pop(word) if word in positive_fd else None
        neutral_fd.pop(word) if word in neutral_fd else None

n = 100  # really trying not to call this variable n_words
print(f"Top {n} most common words in each corpus:")
top_n_positive = {word: count for word, count in positive_fd.most_common(n)}
top_n_neutral = {word: count for word, count in neutral_fd.most_common(n)}
top_n_negative = {word: count for word, count in negative_fd.most_common(n)}
print(f"\tPositive: {top_n_positive}")
print(f"\tNeutral: {top_n_neutral}")
print(f"\tNegative: {top_n_negative}")


def extract_features(text):
    features = {word: 0 for word in
                list(top_n_positive.keys()) + list(top_n_neutral.keys()) + list(top_n_negative.keys())}

    if INCLUDE_BIGRAMS:
        top10_bigrams_pos, top10_bigrams_neu, top10_bigrams_neg = get_top10_bigrams()
        for bigram in (*top10_bigrams_pos, *top10_bigrams_neu, *top10_bigrams_neg):
            features[bigram] = 0


    for sentence in nltk.sent_tokenize(text):
        tokens = nltk.word_tokenize(sentence)
        for word in tokens:
            word = word.lower()
            if word in top_n_positive:
                features[word] += 1
            if word in top_n_neutral:
                features[word] += 1
            if word in top_n_negative:
                features[word] += 1
        if INCLUDE_BIGRAMS:
            for left, right in nltk.bigrams(tokens):
                bigram = (left.lower(), right.lower())
                if bigram in top10_bigrams_pos:
                    features[bigram] += 1
                if bigram in top10_bigrams_neu:
                    features[bigram] += 1
                if bigram in top10_bigrams_neg:
                    features[bigram] += 1

    return features



train_data = []
print("Extracting features for train data...")
with open(TRAIN_PATH, 'r', encoding='utf-8') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
            star_rating = row[1]
            evaluate = extract_features(row[0])
            if evaluate != None:
                if star_rating == "0":
                    train_data.append((list(evaluate.values()), 0))
                elif star_rating == "1":
                    train_data.append((list(evaluate.values()), 1))
                elif star_rating == "2":
                    train_data.append((list(evaluate.values()), 2))


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


print("Extracting features for test data...")
test_data = []
with open(TEST_PATH, 'r', encoding='utf-8') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
            star_rating = row[1]
            evaluate = extract_features(row[0])
            if evaluate != None:
                if star_rating == "0":
                    test_data.append((list(evaluate.values()), 0))
                elif star_rating == "1":
                    test_data.append((list(evaluate.values()), 1))
                elif star_rating == "2":
                    test_data.append((list(evaluate.values()), 2))


classifiers = {
    # "KNeighborsClassifier": KNeighborsClassifier(),
    # "DecisionTreeClassifier": DecisionTreeClassifier(),

    "RandomForestClassifier": RandomForestClassifier(n_estimators=1000, max_features="log2",
                                                     min_samples_leaf=1, min_samples_split=8,
                                                     max_depth=500),

    # "LogisticRegression": LogisticRegression(max_iter=1000),
    # "MLPClassifier": MLPClassifier(max_iter=1000),
    # "AdaBoostClassifier": AdaBoostClassifier(),
}

print("Begin training and classifying...")
results = []
for name, sklearn_classifier in classifiers.items():
    print(f"Training nad classifying with {name}...\t", end="")

    # data format for fit/predict
    train_feats, train_labels = zip(*train_data)
    test_feats, actual_labels = zip(*test_data)

    # fit/predict
    model = sklearn_classifier.fit(train_feats, train_labels)
    prediction_train = model.predict(train_feats)
    prediction_test = model.predict(test_feats)

    # metrics
    accuracy = accuracy_score(y_true=actual_labels, y_pred=prediction_test)
    report_train = classification_report(y_true=train_labels, y_pred=prediction_train)
    report_test = classification_report(y_true=actual_labels, y_pred=prediction_test)
    matrix_train = confusion_matrix(y_true=train_labels, y_pred=prediction_train)
    matrix_test = confusion_matrix(y_true=actual_labels, y_pred=prediction_test)

    # print accuracy and save results
    print(f"Classified with accuracy {accuracy:.2%}!")
    results.append((name, accuracy, (report_train, report_test), (matrix_train, matrix_test)))

# display results
input("hit enter to display the results...\n")
for name, accuracy, (report_train, report_test), (matrix_train, matrix_test) in results:
    print(f"Classification report for {name} (test dataset):\n{report_test}\n")
    disp = ConfusionMatrixDisplay(matrix_test, display_labels=("POSITIVE", "NEUTRAL", "NEGATIVE")).plot()
    disp.ax_.set_title(f"{name} (test dataset)")
    plt.show()

    print(f"Classification report for {name} (train dataset):\n{report_train}\n")
    disp = ConfusionMatrixDisplay(matrix_train, display_labels=("POSITIVE", "NEUTRAL", "NEGATIVE")).plot()
    disp.ax_.set_title(f"{name} (train dataset)")
    plt.show()
