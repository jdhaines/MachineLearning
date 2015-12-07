# Get the stop words repository

# run these two lines once, and save them to C:\Anaconda2\nltk_data
# import nltk
# nltk.download()

from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# grab all the stopwords
sw = stopwords.words('english')
print('There are a total of {} stopwords'.format(len(sw)))

# Stem all the words
stemmer = SnowballStemmer('english')
print('Responsiveness: {}'.format(stemmer.stem('responsiveness')))
print('Responsivity: {}'.format(stemmer.stem('responsivity')))
print('Unresponsive: {}'.format(stemmer.stem('Unresponsive')))

"""
Order of Operations

Stem -> Bag of Words Representation
"""

"""
TFIDF

Tf = term frequency (like bag of words)
Idf = inverse document frequency (weighting by how often the word occurs)
"""
