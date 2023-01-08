import nltk
import matplotlib.pyplot as plt
from csv import reader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay

import utils
from utils import load_unwanted_words
from commons import (
    FEATURES_TOP_N_COMMON_WORDS, FEATURES_INCLUDE_BIGRAMS,
    Corpus, DataPath, Classifier,
    SENTIMENT_CORPORA)


def test_classifiers():
    positive_words = SENTIMENT_CORPORA.words(Corpus.POSITIVE.value)
    neutral_words = SENTIMENT_CORPORA.words(Corpus.NEUTRAL.value)
    negative_words = SENTIMENT_CORPORA.words(Corpus.NEGATIVE.value)

    print("Filtering unwanted words...")
    unwanted_words_log = set(load_unwanted_words())
    unwanted_positive_words = unwanted_words_log & set(positive_words)
    unwanted_neutral_words = unwanted_words_log & set(neutral_words)
    unwanted_negative_words = unwanted_words_log & set(negative_words)

    positive_words = filter(lambda x: x not in unwanted_positive_words, positive_words)
    neutral_words = filter(lambda x: x not in unwanted_neutral_words, neutral_words)
    negative_words = filter(lambda x: x not in unwanted_negative_words, negative_words)

    print("Creating freq dist for each corpus...")
    positive_fd = nltk.FreqDist([w.lower() for w in positive_words])
    neutral_fd = nltk.FreqDist([w.lower() for w in neutral_words])
    negative_fd = nltk.FreqDist([w.lower() for w in negative_words])

    print("Removing duplicates from dataset...")  # but leaving the most frequent in given set
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

    print(f"Top {FEATURES_TOP_N_COMMON_WORDS} most common words in each corpus:")
    top_n_positive = {word: count for word, count in positive_fd.most_common(FEATURES_TOP_N_COMMON_WORDS)}
    top_n_neutral = {word: count for word, count in neutral_fd.most_common(FEATURES_TOP_N_COMMON_WORDS)}
    top_n_negative = {word: count for word, count in negative_fd.most_common(FEATURES_TOP_N_COMMON_WORDS)}
    top10_bigrams = utils.create_top10_bigrams()
    print(f"\tPositive: {top_n_positive}")
    print(f"\tNeutral: {top_n_neutral}")
    print(f"\tNegative: {top_n_negative}")

    def _extract_features_for_review(text):
        features = {word: 0 for word in
                    list(top_n_positive.keys()) + list(top_n_neutral.keys()) + list(top_n_negative.keys())}

        top10_bigrams_pos, top10_bigrams_neu, top10_bigrams_neg = [], [], []
        if FEATURES_INCLUDE_BIGRAMS:
            top10_bigrams_pos, top10_bigrams_neu, top10_bigrams_neg = top10_bigrams
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

            if FEATURES_INCLUDE_BIGRAMS:
                for left, right in nltk.bigrams(tokens):
                    bigram = (left.lower(), right.lower())
                    if bigram in top10_bigrams_pos:
                        features[bigram] += 1
                    if bigram in top10_bigrams_neu:
                        features[bigram] += 1
                    if bigram in top10_bigrams_neg:
                        features[bigram] += 1
        return features

    def _extract_features_for_dataset(data_path: DataPath):
        data = []
        with open(str(data_path.value), 'r', encoding='utf-8') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            if header is not None:
                for row in csv_reader:
                    star_rating = row[1]
                    evaluate = _extract_features_for_review(row[0])
                    if evaluate is not None:
                        if star_rating == "0":
                            data.append((list(evaluate.values()), 0))
                        elif star_rating == "1":
                            data.append((list(evaluate.values()), 1))
                        elif star_rating == "2":
                            data.append((list(evaluate.values()), 2))
        return data

    print("Extracting features for train data...")
    train_data = _extract_features_for_dataset(data_path=DataPath.TRAIN)
    print("Extracting features for test data...")
    test_data = _extract_features_for_dataset(data_path=DataPath.TEST)

    print("Begin training and classifying...")
    results = []
    for classifier in Classifier:
        name, sklearn_classifier = classifier.name, classifier.value
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
