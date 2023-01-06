from csv import reader
import nltk
from nltk.corpus import PlaintextCorpusReader
import enum


positive_reviews = []
neutral_reviews = []
negative_reviews = []

data_files = ["filtered_data/train.csv", "filtered_data/test.csv"]

for file in data_files:
    # skip first line i.e. read header first and then iterate over each row od csv as a list
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        header = next(csv_reader)
        # Check file as empty
        if header != None:
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

# Save corpora to .txt files

# with open('positive_reviews.txt', 'w') as f:
#     f.write('\n'.join(positive_reviews))


sentiment_corpora = PlaintextCorpusReader('corpora_nlp22z', '.*')

class CorpusType(enum.Enum):
    POSITIVE_CORPUS = "positive_reviews.txt"
    NEUTRAL_CORPUS = "neutral_reviews.txt"
    NEGATIVE_CORPUS = "negative_reviews.txt"


stopwords = nltk.corpus.stopwords.words("german")


def skip_unwanted(word):
    if not word.isalpha() or word.lower() in stopwords:
        return False
    return True


positive_words = [word for word in filter(
    skip_unwanted,
    sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value)
)]

neutral_words = [word for word in filter(
    skip_unwanted,
    sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value)
)]

negative_words = [word for word in filter(
    skip_unwanted,
    sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value)
)]

positive_fd = nltk.FreqDist([w.lower() for w in positive_words])
neutral_fd = nltk.FreqDist([w.lower() for w in neutral_words])
negative_fd = nltk.FreqDist([w.lower() for w in negative_words])

common_set = set(positive_fd).intersection(neutral_fd, negative_fd)

for word in common_set:
    del positive_fd[word]
    del neutral_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_neutral = {word for word, count in neutral_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}


def extract_features(text):
    features = dict()
    wordcount_pos = 0
    wordcount_neu = 0
    wordcount_neg = 0

    try:
        for sentence in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sentence):
                if word.lower() in top_100_positive:
                    wordcount_pos += 1
                elif word.lower() in top_100_neutral:
                    wordcount_neu += 1 
                elif word.lower() in top_100_negative:
                    wordcount_neg += 1

        features["wordcount_pos"] = wordcount_pos
        features["wordcount_neu"] = wordcount_neu
        features["wordcount_neg"] = wordcount_neg

        return features

    except:
        return None


train_data_pos = []
train_data_neu = []
train_data_neg = []

with open('filtered_data/train.csv', 'r') as read_obj:
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



from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


classifiers = {
    "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    "MultinomialNB": MultinomialNB(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(max_iter=1000),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


test_data_pos = []
test_data_neu = []
test_data_neg = []

with open('filtered_data/test.csv', 'r') as read_obj:
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
    classifier = nltk.classify.SklearnClassifier(sklearn_classifier)
    classifier.train(train_data)
    accuracy = nltk.classify.accuracy(classifier, test_data)
    print(F"{accuracy:.2%} - {name}")
