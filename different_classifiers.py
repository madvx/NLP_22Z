import os.path
from csv import reader

import matplotlib.pyplot as plt
import nltk
import numpy
from nltk.corpus import PlaintextCorpusReader
import enum
import spacy
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

BALANCED = False
GENERATE_UNWANTED_WORDS_CACHE = False


class CorpusType(enum.Enum):
    POSITIVE_CORPUS = "positive_reviews_balanced.txt" if BALANCED else "positive_reviews.txt"
    NEUTRAL_CORPUS = "neutral_reviews_balanced.txt" if BALANCED else "neutral_reviews.txt"
    NEGATIVE_CORPUS = "negative_reviews_balanced.txt" if BALANCED else "negative_reviews.txt"


sentiment_corpora = PlaintextCorpusReader('corpora_nlp22z', '.*')
CACHE_PATH = "cache/pos_cache_balanced.txt" if BALANCED else "cache/pos_cache.txt"
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
    # features = {"pos": 0, "neu": 0, "neg": 0}
    features = {word: 0 for word in
                list(top_n_positive.keys()) + list(top_n_neutral.keys()) + list(top_n_negative.keys())}
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if word in top_n_positive:
                features[word] += 1
            if word in top_n_neutral:
                features[word] += 1
            if word in top_n_negative:
                features[word] += 1
    return features



train_data = []
print("Extracting features for train data...")
with open('filtered_data/train.csv', 'r', encoding='utf-8') as read_obj:
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
with open('filtered_data/test.csv', 'r', encoding='utf-8') as read_obj:
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
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}

print("Begin training and classifying...")
results = []
for name, sklearn_classifier in classifiers.items():
    print(f"Training nad classifying with {name}...\t", end="")

    # data format for fit/predict
    train_feats, train_labels = zip(*train_data)
    test_feats, actual_labels = zip(*test_data)

    # fit/predict
    prediction = sklearn_classifier.fit(train_feats, train_labels).predict(test_feats)

    # metrics
    accuracy = accuracy_score(y_true=actual_labels, y_pred=prediction)
    report = classification_report(y_true=actual_labels, y_pred=prediction)
    matrix = confusion_matrix(y_true=actual_labels, y_pred=prediction, normalize="pred") # FIXME normalizowac? w jaki sposob?

    # print accuracy and save results
    print(f"Classified with accuracy {accuracy:.2%}!")
    results.append((name, accuracy, report, matrix))

# display results
input("hit enter to display the results...")
for name, accuracy, report, matrix in results:
    print(f"Classification report for {name}:\n{report}\n")
    disp = ConfusionMatrixDisplay(matrix, display_labels=("POSITIVE", "NEUTRAL", "NEGATIVE")).plot(cmap="Blues")
    disp.ax_.set_title(name)
    plt.show()
