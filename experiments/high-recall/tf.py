import cProfile
import json
import time

import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

N_NEIGHBOURS = 100  # number of similar questions
METRIC = ("euclidean", "cityblock", "cosine")  # metric to be used in KNN


def stem_sentence(sentence: str) -> str:
    """
    Stem the input sentence and return processed sentence

    :param sentence: str
        sentence to be stemmed
    :return:
        stemmed sentence
    """
    stemmer = PorterStemmer()  # TODO: try other stemmer types
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
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)


def check_performance(vectorizer: CountVectorizer, knn: NearestNeighbors, vectorized_questions: csr_matrix) -> float:
    """
    Calculate performance of finding similar questions

    :param vectorizer: CountVectorizer
        term frequency vectorizer
    :param knn: NearestNeighbors
        K-nearest neighbors
    :param vectorized_questions: csr_matrix
        input questions transformed with count vectorizer
    :return:
        score (lesser is better)
    """
    print("Performance check started")
    with open("../../data/test_questions_json.json") as json_file:
        json_data = json.load(json_file)

    test_questions = json_data["question"]
    original = json_data["original"]

    test_questions = np.asarray([stem_sentence(test_question) for test_question in test_questions])
    test_questions = vectorizer.transform(test_questions)
    _, indices = knn.kneighbors(test_questions.toarray())

    original = np.asarray([stem_sentence(orig) for orig in original])
    original = vectorizer.transform(original)
    indices_original = np.where((vectorized_questions.toarray() == original.toarray()[:, None]).all(-1))[1]

    rank = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * N_NEIGHBOURS
    score = (rank.sum() + penalization) / indices_original.shape[0]

    return score


def start_tf() -> None:
    """
    Main logic of the TF module

    :return:
        None
    """
    start_time = time.time()
    df = pd.read_csv("../../data/insurance_qna_dataset.csv", sep="\t")
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2))  # TODO: check other params

    questions = np.unique(df.iloc[:, 0].to_numpy())
    questions = np.asarray([stem_sentence(question) for question in questions])
    vectorized_questions = vectorizer.fit_transform(questions)
    print("TF applied")

    knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric=METRIC[2]).fit(vectorized_questions.toarray())
    print("KNN fitted")

    score = check_performance(vectorizer, knn, vectorized_questions)
    print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    nltk.download("punkt")  # used for tokenization
    nltk.download("wordnet")  # used for lemmatization
    cProfile.run('start_tf()')
