import pandas as pd
import numpy as np
import string
import re
import requests
from bs4 import BeautifulSoup

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#import wikipedia

import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import matplotlib
#matplotlib.use('TkAgg')


import streamlit as st

#===============================================================


#========================= FUNCTIONS ======================================
def parse_article(my_url):
    '''parse page'''
    r = requests.get(my_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    return soup

def make_underscore(my_text):
    '''put underscores between words in text'''
    return '_'.join((my_text).split()) 

def make_url(title):
    '''create wikipedia link'''
    return 'https://en.wikipedia.org/wiki/' + make_underscore(title)    

def decontracted(my_text):
    '''convert contractions with apostrophs into words'''
    
    # specific
    phrase = re.sub(r"won\'t", "will not", my_text)
    phrase = re.sub(r"can\'t", "can not", my_text)

    # general
    phrase = re.sub(r"n\'t", " not", my_text)
    phrase = re.sub(r"\'re", " are", my_text)
    phrase = re.sub(r"\'s", " is", my_text)
    phrase = re.sub(r"\'d", " would", my_text)
    phrase = re.sub(r"\'ll", " will", my_text)
    phrase = re.sub(r"\'t", " not", my_text)
    phrase = re.sub(r"\'ve", " have", my_text)
    phrase = re.sub(r"\'m", " am", my_text)
    
    return my_text

def get_plain_text(my_text):
    '''Extract plain text content from paragraphs'''
    paras = []
    for paragraph in my_text.find_all('p'):
        paras.append(str(paragraph.text))
    return paras

def get_header_text(my_text):
    '''Extract text from paragraph headers'''
    heads = []
    for head in my_text.find_all('span', attrs={'mw-headline'}):
        heads.append(str(head.text))
    return heads

def combine_paras_heads(paras, heads):
    '''combine paragraphs & headers'''
    my_text = [val for pair in zip(paras, heads) for val in pair]
    my_text = ' '.join(my_text)
    return my_text

def clean_foot(my_text):
    '''Drop footnote superscripts in brackets'''
    return re.sub(r"\[.*?\]+", '', my_text)

def repl_new_line(my_text):
    '''Replace '\n' (a new line) with '' and end the string at 1000'''
    return my_text.replace('\n', '')[:-11]

def stop():
    '''create stop words list'''
    stop_words = stopwords.words('english')
    stop_words.append('utc')
    stop_words.append('also')
    stop = set(stop_words + list(string.punctuation))
    return stop

def lemm(my_list):
    '''lemmatize both nouns and adjectives  '''
    lemmatizer = WordNetLemmatizer()
    res=[]
    res = [lemmatizer.lemmatize(i) for i in my_list] # lemmatization for nouns
    res = [lemmatizer.lemmatize(i, pos='a') for i in res] # lemmatization for adjectives    
    return res

def stem_title(title):
    '''drop flexions and create list of words in title'''
    stemmer = PorterStemmer()
    return stemmer.stem(title).split()

def exclude(title):
    '''create list of words to be excluded from the resutling list'''
    return set(title.split() + stem_title(title))

# def modals(my_list):
#     '''  '''
#     tagged = nltk.pos_tag(my_list)
#     return set([i for i,j in tagged if j=='MD'])

#========================= SET LAYOUTS ======================================

st.set_page_config(layout="wide")

#============================================================================
st.markdown("# Guess Wiki Page ")

wiki_prefix = 'https://en.wikipedia.org/wiki/'

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    article_title = st.text_input("Enter the article title:", key="my_url", value = 'star wars')

with col2:
    n_show = st.number_input('Number of clue words', min_value=2, max_value=20, value=4)
#============================================================================

modal_verbs = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']

my_url = make_url(article_title)
raw_text = parse_article(my_url)
plain_text = get_plain_text(raw_text)
header_text = get_header_text(raw_text)
my_text = combine_paras_heads(plain_text, header_text)
my_text = clean_foot(repl_new_line(my_text))
my_text = word_tokenize(my_text.lower())

if len(my_text) < 100:
    st.error('Please try another article')
    st.stop()

stop_words = list(stop())

# #remove stop words, punctuation, numbers, uppercase and modal verbs
bw = [decontracted(i) for i in my_text if i.isalpha() and not i in (modal_verbs + stop_words)]
bw = lemm(bw)
bw = [i for i, j in nltk.pos_tag(bw) if j in ['NN', 'NNP']] # only nouns left (arguable). Otherwise you get france/french, japan/japanese etc in clue words. However, we lose 50% of dataset
rstr = '|'.join(exclude(article_title)) # condition for contains

df = pd.DataFrame(bw, columns=['word']).groupby('word').size().sort_values(ascending=False)
df = df[~df.isin(df[df.index.str.contains(rstr)])]



left_col, mid_col, right_col = st.columns(3)
with left_col:
    fig, ax = plt.subplots()
    ax = squarify.plot(sizes=df.head(n_show), label=df.index, color=sns.color_palette('Set3', n_show))
    plt.axis('off')
    st.pyplot(fig)

st.write('Article url:', my_url)

