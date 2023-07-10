import cProfile
import json
import time

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from preprocessing import stem_sentence, lemmatize_sentence

N_NEIGHBOURS = 100  # number of similar questions
METRIC = ("euclidean", "cityblock", "cosine")  # metric to be used in KNN


def check_performance(
    vectorizer: CountVectorizer, knn: NearestNeighbors, vectorized_questions: np.ndarray
) -> float:
    """
    Calculate performance of finding similar questions

    :param vectorizer: CountVectorizer
        term frequency vectorizer
    :param knn: NearestNeighbors
        K-nearest neighbors
    :param vectorized_questions: np.ndarray
        input questions transformed with count vectorizer
    :return:
        score (lesser is better)
    """
    print("Performance check started")
    with open("../../data/test_questions_json.json") as json_file:
        json_data = json.load(json_file)

    test_questions = json_data["question"]
    original = json_data["original"]

    test_questions = np.asarray(
        [stem_sentence(test_question) for test_question in test_questions]
    )
    test_questions = vectorizer.transform(test_questions)
    _, indices = knn.kneighbors(test_questions.toarray())

    original = np.asarray([stem_sentence(orig) for orig in original])
    original = vectorizer.transform(original)
    indices_original = np.where(
        (vectorized_questions == original.toarray()[:, None]).all(-1)
    )[1]

    rank = np.where(indices == indices_original[:, None])[1]
    penalization = (indices_original.shape[0] - rank.shape[0]) * 2 * knn.n_neighbors
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

    vectorizer = CountVectorizer(
        lowercase=True, ngram_range=(1, 1), stop_words="english"
    )

    questions = df.iloc[:, 0].to_numpy()
    questions = np.asarray([stem_sentence(question) for question in questions])
    vectorized_questions = vectorizer.fit_transform(questions)
    vectorized_questions = np.unique(vectorized_questions.toarray(), axis=0)
    print("TF applied")

    knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric=METRIC[2]).fit(
        vectorized_questions
    )
    print("KNN fitted")

    score = check_performance(vectorizer, knn, vectorized_questions)
    print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    nltk.download("punkt")  # used for tokenization
    nltk.download("wordnet")  # used for lemmatization
    # cProfile.run('start_tf()')
    start_tf()
