import json
from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocessing import stem_document, lemmatize_document


def preprocess_documents(
    documents: Iterable[str], preprocessing: str, stemmer: str = "porter"
) -> Iterable[str]:
    """
    Applies preprocessing to iterable of documents

    :param documents: Iterable[str]
        iterable of documents
    :param preprocessing: str
        represents type of preprocessing applied to documents
    :param stemmer: str
        stemmer type
    :return:
        processed iterable of documents
    """
    if preprocessing == "stemming":
        documents = np.asarray(
            [stem_document(document, stemmer) for document in documents]
        )
    elif preprocessing == "lemmatization":
        documents = np.asarray([lemmatize_document(document) for document in documents])

    return documents


def check_performance(
    vectorizer: CountVectorizer,
    knn: NearestNeighbors,
    vectorized_questions: np.ndarray,
    preprocessing: str = None,
    stemmer: str = "porter",
) -> float:
    """
    Calculate performance of finding similar questions

    :param vectorizer: CountVectorizer
        term frequency vectorizer
    :param knn: NearestNeighbors
        K-nearest neighbors
    :param vectorized_questions: np.ndarray
        input questions transformed with count vectorizer
    :param preprocessing: str
        represents type of preprocessing applied to documents
    :param stemmer: str
        stemmer type
    :return:
        score (lesser is better)
    """
    print("Performance check started")
    with open("../../data/test_questions_json.json") as json_file:
        json_data = json.load(json_file)

    test_questions = json_data["question"]
    original = json_data["original"]

    test_questions = preprocess_documents(test_questions, preprocessing, stemmer)
    test_questions = vectorizer.transform(test_questions)
    _, indices = knn.kneighbors(test_questions.toarray())

    original = preprocess_documents(original, preprocessing, stemmer)
    original = vectorizer.transform(original)
    indices_original = np.where(
        (vectorized_questions == original.toarray()[:, None]).all(-1)
    )[1]

    rank = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
    score = (rank.sum() + penalization) / indices_original.shape[0]

    return score
