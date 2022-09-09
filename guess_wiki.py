import pandas as pd
import numpy as np
import string
import re
import requests
from bs4 import BeautifulSoup

import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
import streamlit as st

#========================= SET LAYOUTS ======================================

st.set_page_config(layout="wide")

#============================================================================
st.markdown("# Guess Wiki Page ")
left_column = st.columns([1])

my_url = st.text_input("enter url", key="my_url", value = 'https://en.wikipedia.org/wiki/Jupiter')





# https://levelup.gitconnected.com/two-simple-ways-to-scrape-text-from-wikipedia-in-python-9ce07426579b

#my_url = 'https://en.wikipedia.org/wiki/Russia' 
r = requests.get(my_url)
soup = BeautifulSoup(r.text, 'html.parser')

# Extract the plain text content from paragraphs
paras = []
for paragraph in soup.find_all('p'):
    paras.append(str(paragraph.text))

# Extract text from paragraph headers
heads = []
for head in soup.find_all('span', attrs={'mw-headline'}):
    heads.append(str(head.text))

# Interleave paragraphs & headers
my_text = [val for pair in zip(paras, heads) for val in pair]
my_text = ' '.join(my_text)

# Drop footnote superscripts in brackets
my_text = re.sub(r"\[.*?\]+", '', my_text)

# Replace '\n' (a new line) with '' and end the string at $1000.
my_text = my_text.replace('\n', '')[:-11]

# NLTK
bw = nltk.word_tokenize(my_text)

def decontracted(text):
    '''convert contractions with apostrophs into words'''
    
    # specific
    phrase = re.sub(r"won\'t", "will not", text)
    phrase = re.sub(r"can\'t", "can not", text)

    # general
    phrase = re.sub(r"n\'t", " not", text)
    phrase = re.sub(r"\'re", " are", text)
    phrase = re.sub(r"\'s", " is", text)
    phrase = re.sub(r"\'d", " would", text)
    phrase = re.sub(r"\'ll", " will", text)
    phrase = re.sub(r"\'t", " not", text)
    phrase = re.sub(r"\'ve", " have", text)
    phrase = re.sub(r"\'m", " am", text)
    
    return text

stop = set(stopwords.words('english') + list(string.punctuation))

#remove stop words, punctuation, numbers, uppercase
bw = [decontracted(i) for i in word_tokenize(my_text.lower()) if not i in stop and i.isalpha()]

lemmatizer = WordNetLemmatizer()
bw = [lemmatizer.lemmatize(i) for i in bw] # lemmatization for nouns
bw = [lemmatizer.lemmatize(i, pos='a') for i in bw] # lemmatization for adjectives

df = pd.DataFrame(bw, columns=['word']).groupby('word').size().sort_values(ascending=False)

st.markdown(df.index[1])
