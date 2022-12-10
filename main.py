from typing import Iterable

from ordered_set import OrderedSet

import file_manager
from amazon_scrapping import get_scrapped_reviews, filter_and_format_reviews
from consts import ProductType, Sentiment


def split_reviews_by_sentiment(reviews_with_labels: Iterable[tuple[str, Sentiment]]):
    negative, neutral, positive = OrderedSet(), OrderedSet(), OrderedSet()
    for review_text, review_sentiment in reviews_with_labels:
        if review_sentiment == Sentiment.POSITIVE.value:
            positive.add((review_text, review_sentiment))
        elif review_sentiment == Sentiment.NEUTRAL.value:
            neutral.add((review_text, review_sentiment))
        elif review_sentiment == Sentiment.NEGATIVE.value:
            negative.add((review_text, review_sentiment))
        else:
            print("????")
    return positive, neutral, negative

def print_reviews_statistics(reviews_with_labels: Iterable[tuple[str, Sentiment]]):
    count = {s.value: 0 for s in Sentiment}
    for review_text, review_sentiment in reviews_with_labels:
        count[review_sentiment] += 1
    for sentiment, _count in count.items():
        print(f"{_count} reviews with sentiment '{sentiment}'")
    print(f"Overall {sum(count.values())} reviews")


def print_check_config_for_duplicates():
    products = file_manager.get_config(section="products")
    for product_type in ProductType:
        all_listed_products = list(products[product_type.value])
        dupes = set([product for product in all_listed_products if all_listed_products.count(product) >= 2])
        if dupes:
            print(f"\nWARNING: Found duplicates for Product Type '{product_type.value}': ")
            for dupe in dupes:
                print(f"\t{dupe}")
            print()



def main():

    # check config for duplicates
    print_check_config_for_duplicates()

    # get all reviews for every product
    reviews_with_labels: OrderedSet[tuple[str, Sentiment]] = OrderedSet()

    for product in ProductType:
        # GET RAW, NON-FORMATTED REVIEWS
        raw_reviews: OrderedSet[tuple[str, str, str]] = get_scrapped_reviews(product_type=product,
                                                                             inout_folder="scrapped_data")

        # FILTER REVIEWS (e.g. by language) AND FORMAT TO (text, label)
        filtered_reviews: OrderedSet[tuple[str, Sentiment]] = filter_and_format_reviews(raw_reviews=raw_reviews,
                                                                                        suppress_errors=True)

        reviews_with_labels.update(filtered_reviews)

    print(f"TOTAL:")
    print_reviews_statistics(reviews_with_labels)

    positive, neutral, negative = split_reviews_by_sentiment(reviews_with_labels)
    positive, neutral, negative = positive[:2300], neutral[:2300], negative[:2300]
    print(f"\nBALANCED:")
    print_reviews_statistics([*positive, *neutral, *negative])


if __name__ == "__main__":
    main()
