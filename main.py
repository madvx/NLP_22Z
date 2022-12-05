from amazon_scrapping import get_scrapped_reviews, filter_and_format_reviews
from consts import ProductType, Sentiment
import torch

def main():

    # get all reviews for every product
    reviews_with_labels: list[tuple[str, Sentiment]] = []

    for product in ProductType:
        raw_reviews = get_scrapped_reviews(product_type=product, inout_folder="scrapped_data")
        reviews_with_labels.extend(filter_and_format_reviews(raw_reviews=raw_reviews, suppress_errors=True))

    for review_text, review_sentiment in reviews_with_labels[:3]:
        print(f"\nREVIEW:\n{review_text}\n\nRESULT: {review_sentiment}")


if __name__ == "__main__":
    main()
