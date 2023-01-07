import ast
import itertools
import os
import re

import requests
import file_manager

from bs4 import BeautifulSoup
from typing import Sequence, Iterable
from ordered_set import OrderedSet
from consts import ProductType, Sentiment
import fasttext
model = fasttext.load_model(file_manager.get_filepath("lang_detect_model/lid.176.ftz"))


# LOAD CONFIG
SCRAPPING_CONFIG = file_manager.get_config(section="scrapping")
PRODUCTS = file_manager.get_config(section="products")
#


def __get_url_soup(url: str) -> BeautifulSoup:
    """ Parses HTML webpage at given `url` into searchable BeautifulSoup object """
    headers = {'User-Agent':
               'Mozilla/5.0 (Windows NT 10.0; Win64; x64) \
                AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/90.0.4430.212 Safari/537.36',
               'Accept-Language': 'en-US, en;q=0.5'}
    htmldata = requests.get(url, headers=headers).text
    soup = BeautifulSoup(htmldata, 'html.parser')
    return soup


def __amazon_get_raw_reviews(product_id: str, page_number: int = 1) -> set[tuple[str, str, str]]:
    """ Scrappes author, stars rating and review data for given `product_id` and `page_number`
    returns raw data in form of set of tuples, e.g. { ("Grzegorz", "1.0 out of 5 stars", "Suabe"), ... }
    """
    # PREPARE URL, GET
    if page_number == 0:
        raise Exception("Page numbers start from 1 on amazon")
    url = SCRAPPING_CONFIG["reviews_url"].format(product_id=product_id, page_number=page_number)
    print(url)
    soup = __get_url_soup(url)
    reviews = OrderedSet()

    # TAG AND CLASS NAME OF DATA TO FIND AND SCRAP FROM AMAZON (found using inspect)
    author_tag, author_class_ = "span", "a-profile-name"
    stars_tag, stars_class_ = "span", "a-icon-alt"
    review_tag, review_class_ = "span", "a-size-base review-text review-text-content"

    whole_reviews = soup.find_all("div", id=re.compile("customer_review"))
    for i, review in enumerate(whole_reviews):
        author_text = review.find_next(author_tag, class_=author_class_).get_text()
        starts_text = review.find_next(stars_tag, class_=stars_class_).get_text()
        review_text = review.find_next(review_tag, class_=review_class_).get_text()
        if "The media could not be loaded." in review_text:
            review_text = review_text.replace("The media could not be loaded.", "")
        reviews.add((author_text, starts_text, review_text))
    return reviews


def __amazon_stars_str_to_sentiment(stars_str: str) -> Sentiment:
    """ parses `stars_str` of format "X.X starts out of 5" into "NEGATIVE"/"NEUTRAL"/"POSITIVE" str """
    try:
        rating = stars_str[:4].replace(",", ".")
        star_rating = float(rating)
    except Exception as error:
        print(f"ERROR PARSING START RATING, {stars_str=}")
        raise error

    (min_neg, max_neg), (min_neut, max_neut), (min_pos, max_pos) = SCRAPPING_CONFIG["sentiment_ranges"]
    if min_neg <= star_rating < max_neg:
        return Sentiment.NEGATIVE
    elif min_neut <= star_rating < max_neut:
        return Sentiment.NEUTRAL
    elif min_pos <= star_rating <= max_pos:
        return Sentiment.POSITIVE
    else:
        raise Exception("Bad starts rating")


def __save_raw_reviews_to_file(relative_path: str, raw_reviews: set[tuple[str, str, str]]):
    abs_path = file_manager.get_filepath(file_name=relative_path)
    with open(abs_path, "w+", encoding="utf-8") as file:
        for raw_review in raw_reviews:
            file.write(str(raw_review) + "\n")


def __open_raw_reviews_from_file(relative_path: str) -> OrderedSet[tuple[str, str, str]]:
    abs_path = file_manager.get_filepath(file_name=relative_path)
    raw_reviews = OrderedSet()
    with open(abs_path, "r", encoding="utf-8") as file:
        for line in file.readlines():
            try:
                raw_review = ast.literal_eval(line.rstrip())
                raw_reviews.add(raw_review)
            except Exception as error:
                print(f"Error parsing line: '{line}'!")
                raise error
    return raw_reviews


def __amazon_raw_web_scrap_to_file(product_type: ProductType, product_ids: Iterable[str], out_folder: str = "scrapped_data"):
    """ web scraps raw reviews from amazon into file
    if product was already scrapped before, it'll be skipped
    """
    failed_product_ids: list[str] = []
    product_name = product_type.value


    # iterates for every product of specified type
    for product_id in product_ids:
        print(f"Begin web scrapping for product_id {product_id} (product type: {product_name})...\n"
              f"URL={SCRAPPING_CONFIG['reviews_url'].format(product_id=product_id, page_number='1')}")

        all_raw_reviews = set()
        # iterates until there are no found reviews on page
        for page in itertools.count(start=1):
            print(f"Scrapping page no. {page} for product_id {product_id} (product type: {product_name})...")
            raw_reviews_for_page = __amazon_get_raw_reviews(product_id=product_id, page_number=page)
            if not raw_reviews_for_page:
                if page == 1:
                    failed_product_ids.append(product_id)
                break
            print(f"Scrapped {len(raw_reviews_for_page)} reviews on page no. {page}\n")
            all_raw_reviews.update(raw_reviews_for_page)

            if all_raw_reviews:
                __save_raw_reviews_to_file(relative_path=f"{out_folder}/{product_name}/{product_id}.txt",
                                           raw_reviews=all_raw_reviews)
        print(f"Scrapped all reviews for product_id: {product_id}  (product_type: {product_name})\n")

    if product_ids:
        print(f"Scrapped reviews for products: {product_ids} (product_type: {product_name})")
    if failed_product_ids:
        print(f"WARNING: NO REVIEWS FOR PRODUCTS:")
        for product in failed_product_ids:
            print(f"\t{product}")
        print("REMOVE THEM FROM THE LIST!\n")

    return failed_product_ids


def __format_raw_review_text(raw_review_text: str) -> str:
    # TODO
    return raw_review_text


def __is_review_text_german(review_text: str, suppress_error: bool = False) -> bool:
    """ Returns `True` if text is detected as german. Otherwise, returns False. """
    global model
    try:
        #if detect(review_text) == "de":
        if model.predict(review_text.replace("\n", ""), k=1)[0][0] == "__label__de":
            return True
    except Exception as error:
        if not suppress_error:
            print(f"Error occured during detecting language for review: {review_text}. [{error}]")
    return False


def filter_and_format_reviews(raw_reviews: Sequence[tuple[str, str, str]],
                              suppress_errors: bool = False) -> OrderedSet[tuple[str, Sentiment]]:
    """ filters reviews to only be in proper german lanugage, converts start rating to Sentiment value """
    reviews = OrderedSet()
    for raw_review in raw_reviews:
        author, stars_str, raw_review_text = raw_review

        review_text = __format_raw_review_text(raw_review_text=raw_review_text)
        if not __is_review_text_german(review_text=review_text, suppress_error=suppress_errors):
            continue

        sentiment = __amazon_stars_str_to_sentiment(stars_str=stars_str)

        review_text = review_text.replace("\n", "")
        reviews.add((review_text, sentiment.value))

    return reviews


def get_scrapped_reviews(product_type: ProductType, inout_folder: str = "scrapped_data"
                         ) -> OrderedSet[tuple[str, str, str]]:
    """ Gets scrapped raw review data for every `product_id` in config file of specified `ProductType`.
    If data was previously scrapped for given `product_id` - i.e. file containing raw reviews exists for `product_id` -
    it will be read from that file instead.
    """

    # get all product_ids, for which review data must be scrapped
    all_product_ids: OrderedSet = OrderedSet(PRODUCTS[product_type.value])

    # get all product_ids, for which review data is already scrapped and cached
    product_reviews_data_path = file_manager.get_filepath(f"{inout_folder}/{product_type.value}")
    files = [f for f in os.listdir(product_reviews_data_path) if os.path.isfile(f"{product_reviews_data_path}/{f}")]
    done_product_ids = set([f[:-4] for f in files if f[-4:] == ".txt"])

    # remove already done from all products
    todo_product_ids = set(all_product_ids) - done_product_ids

    # make new scrapped data for todo_product_ids
    failed_product_ids = __amazon_raw_web_scrap_to_file(product_type=product_type, product_ids=todo_product_ids, out_folder=inout_folder)

    all_product_ids -= OrderedSet(failed_product_ids)

    # get and return raw data for all products of given type
    all_reviews = OrderedSet()
    for product_id in all_product_ids:
        reviews = __open_raw_reviews_from_file(relative_path=f"{inout_folder}/{product_type.value}/{product_id}.txt")
        all_reviews.update(reviews)
    return all_reviews





if __name__ == "__main__":


    a= __amazon_get_raw_reviews(product_id="B09GjYPKWV", page_number=41)
    for b in a:
        print(b)

    # # THIS PRINTS QUOTED AMAZON PRODUCT IDS SEPARATED BY COMMA (for config.ini)
    #
    # urls_str = """
    # https://www.amazon.de/Emsa-515615-Isolierbecher-genie%C3%9Fen-Verschluss/dp/B00VE3I2UW/ref=sr_1_5?keywords=Thermobecher&qid=1670686172&sr=8-5
    # https://www.amazon.de/Thermobecher-Ersatzdeckel-Isolierbecher-Thermoisolierbecher-Kaffeebecher/dp/B0792S8YK7/ref=sr_1_6?keywords=Thermobecher&qid=1670686172&sr=8-6
    # """
    #
    # urls = urls_str.split('\n')
    # ids = []
    #
    # for url in urls:
    #     _b = url.find("dp/")
    #     _id = url[_b:]
    #     _id = _id[3:_id.rfind("/")]
    #     print(f""""{_id}", """, end="")

