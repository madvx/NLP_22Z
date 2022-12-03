import ast
from typing import Literal

import requests
from bs4 import BeautifulSoup
import file_manager

scrapping_config = file_manager.get_config(section="scrapping")
products = file_manager.get_config(section="products")


def get_url_soup(url: str) -> BeautifulSoup:
    """ Parses HTML webpage at given `url` into searchable BeautifulSoup object """
    headers = {'User-Agent':
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/90.0.4430.212 Safari/537.36',
               'Accept-Language': 'en-US, en;q=0.5'}
    htmldata = requests.get(url, headers=headers).text
    soup = BeautifulSoup(htmldata, 'html.parser')
    return soup


def amazon_get_raw_reviews(product_id: str, page_number: int = 1) -> set[tuple[str, str, str]]:
    """ Scrappes author, stars rating and review data for given `product_id` and `page_number`
    returns raw data in form of set of tuples, e.g. { ("Grzegorz", "1.0 out of 5 stars", "Suabe"), ... }
    """
    # PREPARE URL, GET
    if page_number == 0:
        raise Exception("Page numbers start from 1 on amazon")
    url = scrapping_config["reviews_url"].format(product_id=product_id, page_number=page_number)
    soup = get_url_soup(url)
    reviews = set()

    # TAG AND CLASS NAME OF DATA TO FIND AND SCRAP FROM AMAZON (found using inspect)
    author_tag, author_class_ = "span", "a-profile-name"
    stars_tag, stars_class_ = "span", "a-icon-alt"
    review_tag, review_class_ = "span", "a-size-base review-text review-text-content"

    skip = 2  # first two reviews are always the same and taken from the rest
    for i, (author, stars, review) in enumerate(zip(soup.find_all(author_tag, class_=author_class_),
                                                    soup.find_all(stars_tag, class_=stars_class_),
                                                    soup.find_all(review_tag, class_=review_class_))):
        if i < skip:
            continue
        review_data = (author.get_text(), stars.get_text(), review.get_text())
        reviews.add(review_data)
    return reviews


def amazon_stars_str_to_sentiment(stars_str: str) -> Literal["NEGATIVE", "NEUTRAL", "POSITIVE"]:
    """ parses `stars_str` of format "X.X starts out of 5" into "NEGATIVE"/"NEUTRAL"/"POSITIVE" str """
    try:
        rating = stars_str[:4].replace(",", ".")
        star_rating = float(rating)
    except Exception as error:
        print(f"ERROR PARSING START RATING, {stars_str=}")
        raise error

    (min_neg, max_neg), (min_neut, max_neut), (min_pos, max_pos) = amazon_sentiment_ranges
    if min_neg <= star_rating < max_neg:
        return "NEGATIVE"
    elif min_neut <= star_rating < max_neut:
        return "NEUTRAL"
    elif min_pos <= star_rating <= max_pos:
        return "POSITIVE"
    else:
        raise Exception("Bad starts rating")


def save_raw_reviews_to_file(relative_path: str, raw_reviews: set[tuple[str, str, str]]):
    abs_path = file_manager.get_filepath(file_name=relative_path)
    with open(abs_path, "w+", encoding="utf-8") as file:
        for raw_review in raw_reviews:
            file.write(str(raw_review) + "\n")


def open_raw_reviews_from_file(relative_path: str) -> set[tuple[str, str, str]]:
    abs_path = file_manager.get_filepath(file_name=relative_path)
    raw_reviews = set()
    with open(abs_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            try:
                raw_review = ast.literal_eval(line.rstrip())
                raw_reviews.add(raw_review)
            except Exception as error:
                print(f"Error parsing line: '{line}'!")
                raise error
    return raw_reviews



all_reviews = set()
product = products["ariel"]
for page in range(1, 20):
    print(f"Scrapping page no. {page}")
    reviews_for_page = amazon_get_raw_reviews(product_id=product, page_number=page)
    print(f"Scrapped {len(reviews_for_page)} reviews on page no. {page}")
    all_reviews.update(reviews_for_page)


print(f"Before saving: {all_reviews=}")
save_raw_reviews_to_file(relative_path=f"scrapped_data/amazon_data_product_{product}.txt", raw_reviews=all_reviews)

all_reviews = open_raw_reviews_from_file(relative_path=f"scrapped_data/amazon_data_product_{product}.txt")
print(f"After saving: {all_reviews=}")
# print([x for x in all_reviews if x[1] == "NEGATIVE"])
