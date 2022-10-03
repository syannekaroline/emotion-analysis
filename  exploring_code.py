from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from collections import Counter
from unicodedata import normalize

import pandas as pd
import nltk
import numpy as np
import string
import emotion_analysis
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
nltk.download('rslp')
nltk.download('stopwords')
plt.style.use('seaborn')

df = emotion_analysis.open_dataset('dataset.xlsx', 'xlsx')
df.Emoção = df.Emoção.str.lower()

print("\033[33m \nAntes do pré-processamnto \033[m \n")
print(df.head())

df.Comentarios = df.Comentarios.apply(emotion_analysis.remove_characters)
print("\n\033[33mApós a remoção de caracteres do string punctuation \033[m \n")
print(df.head())
df.Comentarios = df.Comentarios.apply(emotion_analysis.remove_accents)
print("\n\033[33mApós a remoção de acentos \033[m \n")
print(df.head())
df.Comentarios = df.Comentarios.apply(emotion_analysis.tokenize)
print("\n\033[33mApós a tokentização \033[m \n")
print(df.head())
df.Comentarios = df.Comentarios.apply(emotion_analysis.remove_stopwords)
print("\n\033[33mApós a remoção das stopwords \033[m \n")
print(df.head())
df.Comentarios = df.Comentarios.apply(emotion_analysis.untokenize)
print("\n\033[33mApós a destokentização \033[m \n")
print(df.head())
df.Comentarios = df.Comentarios.apply(emotion_analysis.stemming)
print("\n\033[33mApós a stemmentização \033[m \n")
print(df.head())