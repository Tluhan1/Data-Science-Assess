import sys
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

args = sys.argv[1:]
print(args,"\n")

def Out_Vec_Devec(y1,mode):
    if mode == 0:
        y = []
        for i in y1:
            if i == 'English':
                y.append(0)
            if i == 'Afrikaans':
                y.append(1)
            if i == 'Nederlands':
                y.append(2)
    elif mode == 1:
        y = []
        for i in y1:
            if i == 0:
                y.append('English')
            if i == 1:
                y.append('Afrikaans')
            if i == 2:
                y.append('Nederlands')
    return y


# Main:
text = args

classifier = pickle.load(open('Language_classifier.sav','rb'))
tfidf_vectorizer = pickle.load(open('Vectorizer.sav','rb'))

X = tfidf_vectorizer.transform(text)
Language = classifier.predict(X)
print("\n\nLanguge of the text: ",Out_Vec_Devec(Language,1))