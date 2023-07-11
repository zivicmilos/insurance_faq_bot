from nltk.stem import (
    PorterStemmer,
    SnowballStemmer,
    LancasterStemmer,
    WordNetLemmatizer,
)
from nltk.tokenize import word_tokenize


def stem_document(document: str, stemmer: str = "porter") -> str:
    """
    Stem the input document and return processed document

    :param document: str
        document to be stemmed
    :param stemmer: str
        stemmer type
    :return:
        stemmed document
    """
    if stemmer == "porter":
        stemmer = PorterStemmer()
    elif stemmer == "snowball":
        stemmer = SnowballStemmer(language="english")
    elif stemmer == "lancaster":
        stemmer = LancasterStemmer()
    else:
        raise ValueError(
            f"Stemmer type '{stemmer}' is not supported. Try with 'porter', 'snowball' or 'lancaster'."
        )
    tokens = word_tokenize(document)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def lemmatize_document(document: str) -> str:
    """
    Lemmatize the input document and return processed document

    :param document: str
        document to be lemmatized
    :return:
        lemmatized document
    """
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(document)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)
