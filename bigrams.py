import nltk
from nltk.corpus import PlaintextCorpusReader
import enum

sentiment_corpora = PlaintextCorpusReader('corpora_nlp22z', '.*')

BALANCED = True

class CorpusType(enum.Enum):
    POSITIVE_CORPUS = "positive_reviews_balanced.txt" if BALANCED else "positive_reviews.txt"
    NEUTRAL_CORPUS = "neutral_reviews_balanced.txt" if BALANCED else "neutral_reviews.txt"
    NEGATIVE_CORPUS = "negative_reviews_balanced.txt" if BALANCED else "negative_reviews.txt"


stopwords = nltk.corpus.stopwords.words("german")

positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in  sentiment_corpora.words(CorpusType.POSITIVE_CORPUS.value)
    if w.isalpha() and w.lower() not in stopwords 
])
neutral_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in  sentiment_corpora.words(CorpusType.NEUTRAL_CORPUS.value)
    if w.isalpha() and w.lower() not in stopwords 
])
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words([
    w for w in sentiment_corpora.words(CorpusType.NEGATIVE_CORPUS.value)
    if w.isalpha() and w.lower() not in stopwords 
])


bigram_pos = [w[0] for w in  positive_bigram_finder.ngram_fd.most_common(100)]
bigram_neu = [w[0] for w in  neutral_bigram_finder.ngram_fd.most_common(100)]
bigram_neg = [w[0] for w in  negative_bigram_finder.ngram_fd.most_common(100)]

bigram_pos_dict = dict(w for w in  positive_bigram_finder.ngram_fd.most_common(100))
bigram_neu_dict = dict(w for w in  neutral_bigram_finder.ngram_fd.most_common(100))
bigram_neg_dict = dict(w for w in  negative_bigram_finder.ngram_fd.most_common(100))

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

print('POSITIVE BIGRAMS: \n', bigram_pos)
print()
print('NEUTRAL BIGRAMS: \n', bigram_neu)
print()
print('NEGATIVE BIGRAMS: \n', bigram_neg)
print()


