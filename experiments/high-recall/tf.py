import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

N = 100  # number of similar questions
metric = ['euclidean', 'cityblock', 'cosine']  # metric to be used in KNN

df = pd.read_csv('../../data/insurance_qna_dataset.csv', sep='\t')
df.drop(columns=df.columns[0], axis=1, inplace=True)
questions = df.iloc[:, 0].to_numpy()
questions = np.unique(questions)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(questions)

new_question = ['What Happens When Life Insurance Is Paid Up?']
new_question = vectorizer.transform(new_question)

knn = NearestNeighbors(n_neighbors=N, metric=metric[2]).fit(X.A)
distances, indices = knn.kneighbors(new_question.A)
