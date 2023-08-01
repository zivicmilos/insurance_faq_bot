import datetime
import itertools
import re
from time import time

import keras.backend as K
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim import downloader
from keras.layers import Input, Embedding, LSTM, Lambda
from keras.models import Model
from keras.optimizers import Adadelta
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download("stopwords")

# File paths
TRAIN_TSV = "../../data/quora_duplicate_questions.tsv"


# Load training and test set
train_df = pd.read_csv(TRAIN_TSV, delimiter="\t").iloc[:, 3:]
test_df = train_df.iloc[:, :2]

stops = set(stopwords.words("english"))


def text_to_word_list(text):
    """Preprocess and convert texts to a list of words"""
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text


# Prepare embedding
vocabulary = dict()
inverse_vocabulary = [
    "<unk>"
]  # '<unk>' will never be used, it is only a placeholder for the [0, 0, ....0] embedding
word2vec = downloader.load("word2vec-google-news-300")

questions_cols = ["question1", "question2"]

# Iterate over the questions only of both training and test datasets
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        # Iterate through the text of both questions of the row
        for question in questions_cols:
            q2n = []  # q2n -> question numbers representation
            for word in text_to_word_list(row[question]):
                # Check for unwanted words
                if word in stops and word not in word2vec.key_to_index:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    q2n.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)
                else:
                    q2n.append(vocabulary[word])

            # Replace questions as word to question as number representation
            # dataset.set_value(index, question, q2n)
            dataset.at[index, question] = q2n
# train_df.to_csv("train_df.csv", sep=",", index=False)
# test_df.to_csv("test_df.csv", sep=",", index=False)

# train_df = pd.read_csv("train_df.csv")
# test_df = pd.read_csv("test_df.csv")

embedding_dim = 300
embeddings = 1 * np.random.randn(
    len(vocabulary) + 1, embedding_dim
)  # This will be the embedding matrix
embeddings[0] = 0  # So that the padding will be ignored

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.key_to_index:
        embeddings[index] = word2vec.get_vector(word)

del word2vec

max_seq_length = max(
    train_df.question1.map(lambda x: len(x)).max(),
    train_df.question2.map(lambda x: len(x)).max(),
    test_df.question1.map(lambda x: len(x)).max(),
    test_df.question2.map(lambda x: len(x)).max(),
)

# Split to train validation
validation_size = 40000
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = train_df["is_duplicate"]

X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, Y, test_size=validation_size
)

# Split to dicts
X_train = {"left": X_train.question1, "right": X_train.question2}
X_validation = {"left": X_validation.question1, "right": X_validation.question2}
X_test = {"left": test_df.question1, "right": test_df.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ["left", "right"]):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Make sure everything is ok
assert X_train["left"].shape == X_train["right"].shape
assert len(X_train["left"]) == len(Y_train)

# Model variables
n_hidden = 50
gradient_clipping_norm = 1.25
batch_size = 64
n_epoch = 25


def exponent_neg_manhattan_distance(left, right):
    """Helper function for the similarity estimate of the LSTMs outputs"""
    return K.exp(-K.sum(K.abs(left - right), axis=1, keepdims=True))


# The visible layer
left_input = Input(shape=(max_seq_length,), dtype="int32")
right_input = Input(shape=(max_seq_length,), dtype="int32")

embedding_layer = Embedding(
    len(embeddings),
    embedding_dim,
    weights=[embeddings],
    input_length=max_seq_length,
    trainable=False,
)

# Embedded version of the inputs
encoded_left = embedding_layer(left_input)
encoded_right = embedding_layer(right_input)

# Since this is a siamese network, both sides share the same LSTM
shared_lstm = LSTM(n_hidden)

left_output = shared_lstm(encoded_left)
right_output = shared_lstm(encoded_right)

# Calculates the distance as defined by the MaLSTM model
malstm_distance = Lambda(
    function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),
    output_shape=lambda x: (x[0][0], 1),
)([left_output, right_output])

# Pack it all up into a model
malstm = Model([left_input, right_input], [malstm_distance])

# Adadelta optimizer, with gradient clipping by norm
optimizer = Adadelta(clipnorm=gradient_clipping_norm)

malstm.compile(loss="mean_squared_error", optimizer=optimizer, metrics=["accuracy"])

# Start training
training_start_time = time()

malstm_trained = malstm.fit(
    [X_train["left"], X_train["right"]],
    Y_train,
    batch_size=batch_size,
    epochs=n_epoch,
    validation_data=([X_validation["left"], X_validation["right"]], Y_validation),
)

print(
    "Training time finished.\n{} epochs in {}".format(
        n_epoch, datetime.timedelta(seconds=time() - training_start_time)
    )
)

# Plot accuracy
plt.plot(malstm_trained.history["acc"])
plt.plot(malstm_trained.history["val_acc"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Plot loss
plt.plot(malstm_trained.history["loss"])
plt.plot(malstm_trained.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper right")
plt.show()
