# NLP 22Z

Sentiment analysis for German language reviews for products:
* Thermobecher    (thermal mug)
* Spülmaschinen-Tabs    (dishwasher tablets)
* Colorwaschmittel    (color washing powder)

### Dependencies and requirements
Required Python version: `Python 3.9`  
All necessary modules are in requirements.txt, install using: `pip3 install -r requirements.txt`

Additionally, it's needed to run within Python console:
> import nltk
> import spacy
> nltk.download([  
>    "names",  
>    "stopwords",  
>    "state_union",  
>    "twitter_samples",  
>    "movie_reviews"  
>    "averaged_perceptron_tagger",  
>    "vader_lexicon",  
>    "punkt",  
> ])
> nlp = spacy.load('de_core_news_sm')

