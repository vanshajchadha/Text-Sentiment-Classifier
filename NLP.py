# NLP: Text Sentiment Classifier from movie reviews

# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import errno

# Importing the dataset

path_positive='/Users/vanshajchadha/Desktop/review_polarity/txt_sentoken/pos/*.txt'
path_negative='/Users/vanshajchadha/Desktop/review_polarity/txt_sentoken/neg/*.txt'

files_pos=glob.glob(path_positive)
files_neg=glob.glob(path_negative)

list_pos=[]

for name in files_pos:
    try:
        with open(name) as f:
            message=f.read()
            list_pos.append(message)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
list_neg=[]

for name in files_neg:
    try:
        with open(name) as f:
            message=f.read()
            list_neg.append(message)
    except IOError as exc:
        if exc.errno != errno.EISDIR:
            raise
                  
dataseries_pos=pd.Series(list_pos,[i for i in range(len(list_pos))])
dataseries_neg=pd.Series(list_neg,[i for i in range(len(list_pos),len(list_pos)+len(list_neg))])

data=dataseries_pos.append(dataseries_neg)

dataset=pd.DataFrame(data,columns=['Values'])

dataset['Classified']=pd.Series(np.ones(1000)).append(pd.Series(np.zeros(1000),[i for i in range(1000,2000)]))

dataset=dataset.sample(frac=1).reset_index(drop=True) # For shuffling the rows in the dataframe

# Cleaning the texts and Removing the words that are insignificant

import re
import nltk
# nltk.download('stopwords')  # List of insignificant words
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # Stemming: Simply keeping the root of the word
from nltk.stem.wordnet import WordNetLemmatizer  # Lemmatizing

# ps=PorterStemmer()
lm=WordNetLemmatizer() 
corpus = []      # Corpus simply means a collection of texts in NLP

for i in range(0,2000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Values'][i])
    # review=review.lower()
    review=review.split()
    
    # Stemming is faster than Lemmatizing as it doesn't care about the context and thus, has lower accuracy
    # review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # Lemmatizing, however, finds a synonym or something meaningful most of the times and its default part of speech is noun
    
    review=[lm.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review =' '.join(review)      # Converting the list back to a string
    corpus.append(review)

# Creating the Bag of Words Model: Tokenization --> Creation of Sparse Matrix with each unique word having a different column

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()    # Max_features limits the number of columns to reduce the sparsity as well as remove words which seldom appear
X=cv.fit_transform(corpus).toarray()  # No header as it is a Matrix and not a Dataframe
y=dataset.iloc[:,-1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





