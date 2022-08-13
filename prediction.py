import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data/clean_kaggle_dataset.csv')

model_path = 'saved_model/finalized_model.sav'
tfidf = TfidfVectorizer(max_features=3000)
tfidf.fit(df['text'])

transform = {
    0: "ham",
    1: "spam"
}


def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return model

model = load_model(model_path)
def predict_txt(text):
    text = np.array([text])
    text = tfidf.transform(text).toarray()
    prediction = model.predict(text)
    prediction = transform[prediction[0]]
    return prediction

