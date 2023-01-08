from csv import reader


positive_reviews = []
neutral_reviews = []
negative_reviews = []

BALANCED = True


TRAIN_PATH = "filtered_data/train_balanced.csv" if BALANCED else "filtered_data/train.csv"
TEST_PATH = "filtered_data/test_balanced.csv" if BALANCED else "filtered_data/test.csv"

POSITIVE_CORPUS = "corpora_nlp22z/positive_reviews_balanced.txt" if BALANCED else "corpora_nlp22z/positive_reviews.txt"
NEUTRAL_CORPUS = "corpora_nlp22z/neutral_reviews_balanced.txt" if BALANCED else "corpora_nlp22z/neutral_reviews.txt"
NEGATIVE_CORPUS = "corpora_nlp22z/negative_reviews_balanced.txt" if BALANCED else "corpora_nlp22z/negative_reviews.txt"


# skip first line i.e. read header first and then iterate over each row od csv as a list
with open(TRAIN_PATH, 'r', encoding='utf-8') as read_obj:
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

with open(POSITIVE_CORPUS, 'w', encoding='utf-8') as f:
    f.write('\n'.join(positive_reviews))

with open(NEUTRAL_CORPUS, 'w', encoding='utf-8') as f:
    f.write('\n'.join(neutral_reviews))

with open(NEGATIVE_CORPUS, 'w', encoding='utf-8') as f:
    f.write('\n'.join(negative_reviews))
