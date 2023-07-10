import cProfile
import time

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

from utils import check_performance, preprocess_documents

N_NEIGHBOURS = 100  # number of similar questions
METRIC = ("euclidean", "cityblock", "cosine")  # metric to be used in KNN
PREPROCESSING = "stemming"
STEMMER = "snowball"


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
    questions = preprocess_documents(questions, PREPROCESSING, STEMMER)
    vectorized_questions = vectorizer.fit_transform(questions)
    vectorized_questions = np.unique(vectorized_questions.toarray(), axis=0)
    print("TF applied")

    knn = NearestNeighbors(n_neighbors=N_NEIGHBOURS, metric=METRIC[2]).fit(
        vectorized_questions
    )
    print("KNN fitted")

    score = check_performance(
        vectorizer, knn, vectorized_questions, PREPROCESSING, STEMMER
    )
    print(f"Score: {score:.2f} | ETA: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    nltk.download("punkt")  # used for tokenization
    nltk.download("wordnet")  # used for lemmatization
    # cProfile.run('start_tf()')
    start_tf()
