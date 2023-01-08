# NLP 22Z

Sentiment analysis for German language reviews for products:
* Thermobecher    (thermal mug)
* Spülmaschinen-Tabs    (dishwasher tablets)
* Colorwaschmittel    (color washing powder)

### Dependencies and requirements
Required Python version: `Python 3.9`  
All necessary modules are in requirements.txt, install using: `pip3 install -r requirements.txt`

Linux user should first run this command:  `python3 -m spacy download de_core_news_sm`  
  
Additionally, it's needed to run within Python console:
> import nltk  
> import spacy  
> nltk.download([  
>    "stopwords",  
>    "averaged_perceptron_tagger",  
>    "punkt",  
> ])  
> nlp = spacy.load('de_core_news_sm')

### Documentation  
Link to Google Docs with final documentation: [LINK](https://docs.google.com/document/d/1FWsgpyKUKIrMwfQum-13OUG3LK8Ly5Dy1hskzDFsiv4/edit#)
