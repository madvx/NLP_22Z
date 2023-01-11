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


### Running
#### Multiple classifiers test
Run classifiers test with command: `python3 main.py`  
Generate corpora with command: `python3 main.py -m gen_corpora`  
Generate unwanted cache (stopwords + nouns) for faster test with command: `python3 main.py -m gen_unwanted_cache`

#### Bert finetuning
1. Upload to google colab:
   1. _bert_config.json_ - provide parameters in this file
   2. _BERT_finetuning.ipynb_ 
   3. datasets in .csv format
2. Run all cells