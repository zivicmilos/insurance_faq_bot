import json
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


def check_performance() -> float:
    """
    Calculate performance of finding similar questions

    :return:
        score (lesser is better)
    """
    with open("../../data/test_questions_json.json") as json_file:
        json_data = json.load(json_file)

    test_questions = json_data["question"]
    original = json_data["original"]

    test_questions = [stem_sentence(test_question) for test_question in test_questions]
    test_questions = vectorizer.transform(test_questions)
    distances, indices = knn.kneighbors(test_questions.A)

    original = [stem_sentence(orig) for orig in original]
    original = vectorizer.transform(original)
    indices_original = np.where((X.A == original.A[:, None]).all(-1))[1]

    x = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - x.shape[0]) * 2 * N
    score = (x.sum() + penalization) / indices_original.shape[0]

    return score


nltk.download("punkt")  # used for tokenization
nltk.download("wordnet")  # used for lemmatization

N = 100  # number of similar questions
metric = ["euclidean", "cityblock", "cosine"]  # metric to be used in KNN
stemmer = PorterStemmer()  # TODO: try other stemmer types
lemmatizer = WordNetLemmatizer()

df = pd.read_csv("../../data/insurance_qna_dataset.csv", sep="\t")
df.drop(columns=df.columns[0], axis=1, inplace=True)

questions = df.iloc[:, 0].to_numpy()
questions = [stem_sentence(question) for question in questions]
questions = np.asarray(questions)
questions = np.unique(questions)

vectorizer = CountVectorizer(lowercase=True)  # TODO: check other params, fix n-gram problem
X = vectorizer.fit_transform(questions)

knn = NearestNeighbors(n_neighbors=N, metric=metric[2]).fit(X.A)

score = check_performance()
print(score)
