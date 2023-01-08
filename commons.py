import enum

import nltk
import spacy
from nltk.corpus import PlaintextCorpusReader
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from file_manager import get_config, get_filepath as path

# load common settings
test_classifiers_settings = get_config(section="test_classifiers")
general_settings = get_config(section="general")

# load consts
BALANCED = general_settings["use_balanced_data"]
FEATURES_TOP_N_COMMON_WORDS = test_classifiers_settings["features_top_n_common_words"]
FEATURES_INCLUDE_BIGRAMS = test_classifiers_settings["features_include_top10_bigrams"]
SENTIMENT_CORPORA = PlaintextCorpusReader(path('corpora_nlp22z'), '.*')
GERMAN_STOPWORDS = nltk.corpus.stopwords.words("german")
GERMAN_SPACY_MODEL = spacy.load('de_core_news_sm')


class ProductType(enum.Enum):
    THERMAL_MUG = "thermal_mug"
    DISHWASHER_TABLETS = "dishwasher_tablets"
    COLOR_WASHING_POWDER = "color_washing_powder"


class Sentiment(enum.Enum):
    POSITIVE = "POSITIVE"
    NEUTRAL = "NEUTRAL"
    NEGATIVE = "NEGATIVE"


class Corpus(enum.Enum):
    POSITIVE = "positive_reviews_balanced.txt" if BALANCED else "positive_reviews.txt"
    NEUTRAL = "neutral_reviews_balanced.txt" if BALANCED else "neutral_reviews.txt"
    NEGATIVE = "negative_reviews_balanced.txt" if BALANCED else "negative_reviews.txt"


class DataPath(enum.Enum):
    TRAIN = path("csv_data/train_balanced.csv") if BALANCED else path("csv_data/train.csv")
    TEST = path("csv_data/test_balanced.csv") if BALANCED else path("csv_data/test.csv")
    CACHE = path("cache/unwanted_words_cache_balanced.txt") if BALANCED else path("cache/unwanted_words_cache.txt")


class Classifier(enum.Enum):
    KNeighborsClassifier = KNeighborsClassifier(**test_classifiers_settings["k_neighbors_kwargs"])
    DecisionTreeClassifier = DecisionTreeClassifier(**test_classifiers_settings["decision_tree_kwargs"])
    RandomForestClassifier = RandomForestClassifier(**test_classifiers_settings["random_forest_kwargs"])
    LogisticRegression = LogisticRegression(**test_classifiers_settings["logistic_regression_kwargs"])
    MLPClassifier = MLPClassifier(**test_classifiers_settings["mlp_kwargs"])
    AdaBoostClassifier = AdaBoostClassifier(**test_classifiers_settings["ada_boost_kwargs"])

