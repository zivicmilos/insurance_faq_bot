from nltk.stem import (
    PorterStemmer,
    SnowballStemmer,
    LancasterStemmer,
    WordNetLemmatizer,
)
from nltk.tokenize import word_tokenize


def stem_sentence(sentence: str) -> str:
    """
    Stem the input sentence and return processed sentence

    :param sentence: str
        sentence to be stemmed
    :return:
        stemmed sentence
    """
    # stemmer = PorterStemmer()
    stemmer = SnowballStemmer("english")
    # stemmer = LancasterStemmer()
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
