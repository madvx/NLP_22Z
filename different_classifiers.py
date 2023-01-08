from csv import reader
import nltk
from nltk.corpus import PlaintextCorpusReader
import enum
import spacy

BALANCED = True
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
    print("pozytywne")
    positive_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)

    print("neutralne")
    neutral_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)

    print("negatywne")
    negative_words = [word for word in filter(
        skip_unwanted,
        set(sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value))
    )]
    save_unwanted_words(unwanted_words_log)
    exit()

else:
    positive_words = filter(lambda x: x not in unwanted_words_log,
                            sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value))
    neutral_words = filter(lambda x: x not in unwanted_words_log,
                           sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value))
    negative_words = filter(lambda x: x not in unwanted_words_log,
                            sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value))

positive_fd = nltk.FreqDist([w.lower() for w in positive_words])
neutral_fd = nltk.FreqDist([w.lower() for w in neutral_words])
negative_fd = nltk.FreqDist([w.lower() for w in negative_words])

# remove duplicates but leaving the most frequent in given set
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
top_100_positive = {word: count for word, count in positive_fd.most_common(10)}
top_100_neutral = {word: count for word, count in neutral_fd.most_common(10)}
top_100_negative = {word: count for word, count in negative_fd.most_common(10)}
print(top_100_positive)
print(top_100_neutral)
print(top_100_negative)


def extract_features(text):
    features = {"pos": 0, "neu": 0, "neg": 0}
    features = {word: 0 for word in
                list(top_100_positive.keys()) + list(top_100_neutral.keys()) + list(top_100_negative.keys())}
    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            word = word.lower()
            if word in top_100_positive:
                features[word] += 1
            if word in top_100_neutral:
                features[word] += 1
            if word in top_100_negative:
                features[word] += 1
    return features


train_data_pos = []
train_data_neu = []
train_data_neg = []

with open('filtered_data/train.csv', 'r', encoding='utf-8') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
            star_rating = row[1]
            if star_rating == "0":
                train_data_pos.append(row[0])
            elif star_rating == "1":
                train_data_neu.append(row[0])
            elif star_rating == "2":
                train_data_neg.append(row[0])

train_data = []

# LABELS  = {"POSITIVE": 0, "NEUTRAL": 1, "NEGATIVE": 2}

for review in train_data_pos:
    evaluate = extract_features(review)
    if evaluate != None:
        train_data.append((evaluate, 0))

for review in train_data_neu:
    evaluate = extract_features(review)
    if evaluate != None:
        train_data.append((evaluate, 1))

for review in train_data_neg:
    evaluate = extract_features(review)
    if evaluate != None:
        train_data.append((evaluate, 2))




from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = {
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


test_data_pos = []
test_data_neu = []
test_data_neg = []

with open('filtered_data/test.csv', 'r', encoding='utf-8') as read_obj:
    csv_reader = reader(read_obj)
    header = next(csv_reader)
    if header != None:
        for row in csv_reader:
            star_rating = row[1]
            if star_rating == "0":
                test_data_pos.append(row[0])
            elif star_rating == "1":
                test_data_neu.append(row[0])
            elif star_rating == "2":
                test_data_neg.append(row[0])

test_data = []

for review in test_data_pos:
    evaluate = extract_features(review)
    if evaluate != None:
        test_data.append((evaluate, 0))

for review in test_data_neu:
    evaluate = extract_features(review)
    if evaluate != None:
        test_data.append((evaluate, 1))

for review in test_data_neg:
    evaluate = extract_features(review)
    if evaluate != None:
        test_data.append((evaluate, 2))

for name, sklearn_classifier in classifiers.items():
    print(f"{name}...\t", end="")
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(train_data)
    accuracy = nltk.classify.accuracy(classifier, test_data)
    print(f"{accuracy:.2%}")
