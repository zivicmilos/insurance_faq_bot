import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


def stem_sentence(sentence: str) -> str:
    """
    Stem the input sentence and return processed sentence

    :param sentence: str
        sentence to be stemmed
    :return:
        stemmed sentence
    """
    tokens = word_tokenize(sentence)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def lemmatize_sentence(sentence: str) -> str:
    """
    Lemmatize the input sentence and return processed sentence

    :param sentence: str
        sentence to be lemmatized
    :return:
        lemmatized sentence
    """
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


nltk.download("punkt")  # used for tokenization
nltk.download("wordnet")  # used for lemmatization

N = 100  # number of similar questions
metric = ["euclidean", "cityblock", "cosine"]  # metric to be used in KNN
stemmer = PorterStemmer()  # TODO: try other stemmer types
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("../../data/insurance_qna_dataset.csv", sep="\t")
df.drop(columns=df.columns[0], axis=1, inplace=True)

questions = df.iloc[:, 0].to_numpy()
questions = [lemmatize_sentence(questions) for questions in questions]
questions = np.asarray(questions)
questions = np.unique(questions)

vectorizer = CountVectorizer(lowercase=True)  # TODO: check other params
X = vectorizer.fit_transform(questions)

new_question = "What Happens When Life Insurance Is Paid Up?"
new_question = lemmatize_sentence(new_question)
new_question = vectorizer.transform([new_question])

knn = NearestNeighbors(n_neighbors=N, metric=metric[2]).fit(X.A)
distances, indices = knn.kneighbors(new_question.A)
