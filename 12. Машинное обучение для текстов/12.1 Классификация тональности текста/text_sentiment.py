import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
nltk.download('stopwords')
stopwords = list(set(nltk_stopwords.words('russian')))
count_tf_idf = TfidfVectorizer(stop_words=stopwords)

train = pd.read_csv('tweets_lemm_train.csv')
test = pd.read_csv('tweets_lemm_test.csv')

X_train = count_tf_idf.fit_transform(train.lemm_text)
y_train = train.positive
X_test = count_tf_idf.transform(test.lemm_text)

model = LogisticRegression(random_state=RANDOM_STATE)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
pd.Series(y_pred, name='positive').to_csv('predictions', index=False)