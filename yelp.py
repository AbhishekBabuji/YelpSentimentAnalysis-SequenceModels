"""
yelp.py

(C) 2018 by Abhishek Babuji <abhishekb2209@gmail.com>

Creates and trains sequence models on yelp-pizza reviews
"""

#pylint: disable=import-error
import json
import re
import string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import gensim
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords



def read_small_dataset(file_path, file_type):

    """
    Only use for small JSON or CSV files. If it is a large dataset, then you'll need
    to appropriately read only specific columns or in chunks to save space.

    Args:
        file_path (str): Path in the local directory
        file_type (str): Can be 'JSON' or 'CSV'


    Returns:
        data_frame (pandas.DataFrame)

    """

    if file_type == 'JSON':
        data_frame = pd.read_json(file_path, lines=True)

    else:
        data_frame = pd.read_csv(file_path)

    return data_frame

def read_large_dataset(file_path, file_type, column_names):


    """

    Args:
        file_path (str): Path in the local directory
        file_type (str): Can be 'JSON' or 'CSV'
        column_names (list): List of columns to be read

    Returns:
        data_frame (pandas.DataFrame)

    """

    empty_list = [] #List to push in all the relevant rows and columns

    if file_type == 'JSON':
        with open(file_path, 'r') as file_opened:
            for line in file_opened:
                data = json.loads(line)
                empty_list.append([data[column_names[0]],
                                   data[column_names[1]],
                                   data[column_names[2]]])

        data_frame = pd.DataFrame(empty_list)
        data_frame.columns = column_names
        return data_frame


    data_frame = pd.read_csv(file_path)
    return data_frame


def clean_text(text):

    """

    Args:
        text (str): Each row of a DataFrame as text

    Returns:
        text (str): cleaned test

    """
    ## Remove puncuation
    text = text.translate(string.punctuation)

    ## Convert words to lower case and split them
    text = text.lower().split()

    ## Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]

    text = " ".join(text)

    ## Clean the text: Self explanatory
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
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

    return text

def create_embedding_index(model_path):

    """

    Args:
        model_path (str): Path to Word2vec model in the local file system

    Returns:
        embedding_index (dict): A dictionary of vectors representing word embedding for each word
    """

    model = gensim.models.KeyedVectors.load_word2vec_format(model_path,
                                                            binary=True)

    words = model.index2word
    embedding_index = dict()

    for word in words:
        embedding_index[word] = model[word]

    return embedding_index

def create_padded_sequence(vocabulary_size, maxlen, data_frame, text_column):

    """

    Args:
        vocabulary_size (int): Top `vocabulary_size` to be considered
        maxlen (int): Maximum length of the sequence
        data_frame (pandas.DataFrame): Yelp pizza DataFrame containing reviews

    Returns:
        data (np.array): Our input data converted to sequence
        tokenizer (keras.preprocessing.text.Tokenizer): Keras tokenizer object

    """


    tokenizer = Tokenizer(num_words=vocabulary_size)
    tokenizer.fit_on_texts(data_frame[text_column])
    sequences = tokenizer.texts_to_sequences(data_frame[text_column])
    data = pad_sequences(sequences, maxlen)

    return data, tokenizer

def create_embedding_matrix(vocabulary_size, num_dimensions, tokenizer, embeddings_index):

    """

    Args:
        vocabulary_size (int): Top `vocabulary_size` words
                               being considered whose word vectors are being extracted

        num_dimensions (int): Word vector dimensions
        tokenizer (keras.preprocessing.text.Tokenizer): Keras tokenizer object
        embeddings_index (dict): Dictionary containing indices of words

    Returns:
        embedding_matrix (np.array): This will be the input

    """


    embedding_matrix = np.zeros((vocabulary_size, num_dimensions))

    for word, index in tokenizer.word_index.items():

        if index > vocabulary_size - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix


def fit_lstms(num_units, embedding_weights, num_epochs, fit_data):

    """

    Args:
        num_units (int): Number of LSTM units
        weights (numpy.array): The embedding matrix
        epochs (int): Number of gradient descent iterations
        fit_data (dict): Dictionary containing the training and testing data

    Returns:
        model_glove (Keras model): Can be used to predict on unseen data

    """
    model_glove = Sequential()
    model_glove.add(Embedding(50000,
                              300,
                              input_length=60,
                              weights=embedding_weights,
                              trainable=False))

    model_glove.add(Dropout(0.5))
    model_glove.add(Conv1D(64, 5, activation='relu'))
    model_glove.add(MaxPooling1D(pool_size=4))
    model_glove.add(Bidirectional(LSTM(num_units)))
    model_glove.add(Dense(3, activation='softmax'))
    model_glove.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model_glove.summary())

    model_glove.fit(fit_data['x_train'],
                    fit_data['y_train'],
                    validation_data=(fit_data['x_test'], fit_data['y_test']),
                    epochs=num_epochs)

    return model_glove

def main():

    """

    Main method divided into parts

    """
    print("Reading in business.json")
    #Part 1.1: Extracting Pizza Restaurants from business.json
    business_file_path = "/Volumes/Elements/December7th/yelp_dataset/yelp_academic_dataset_business.json"
    business = read_small_dataset(business_file_path, 'JSON')
    business.dropna(subset=['categories'], inplace=True) #Drop categories column
    business.isna().sum() #Check number of NaNs in a DataFrame

    pizza = business[business['categories'].str.contains('Pizza')]

    print("Reading in reviews.json")
    #Part 1.2: Extracting Stars and Text Reviews
    yelp_review_path = "/Volumes/Elements/December7th/yelp_dataset/yelp_academic_dataset_review.json"
    yelp_review_column_names = ['stars', 'text', 'business_id']
    reviews = read_large_dataset(yelp_review_path, 'JSON', yelp_review_column_names)
    reviews.columns = ['rating', 'review', 'business_id']

    print("Merging pizza and business DataFrames")
    #Part 1.3: Final Dataset with reviews for Pizza Joints
    yelp_pizza = pd.merge(reviews, pizza, how='inner', on=['business_id'])
    yelp_pizza = yelp_pizza[['stars', 'review']]


    #Part 2.1 Creating smaller number of categories from larger categories and cleaning text
    print("Creating smaller number of categories from larger categories...")

    yelp_pizza['stars'] = yelp_pizza['stars'].\
        apply({1: 'Bad', 2: 'Bad', 3: 'Average', 4: 'Good', 5: 'Good'}.get)
    yelp_pizza.dropna(axis=0, inplace=True)
    print("Done!")
    print()
    print("Cleaning text...")
    print("Done!")
    yelp_pizza['review'] = yelp_pizza['review'].map(lambda x: clean_text(x))
    print()

    #Writing to CSV so we have a local copy, and we can store the cleaned dataset instead of reading it in everytime
    print("Writing to CSV...")
    yelp_pizza.to_csv("/Volumes/Elements/Yelp Pizza/yelp_pizza.csv")
    print("Done!")
    print()

    #Some summary statistics
    print("Printing summary statistics...")
    print("Average number of words by review type (Good, Bad and Average)")
    print(yelp_pizza.groupby('stars').review.apply(lambda x: x.str.split().str.len().mean()))
    print()
    print("Number of datapoints")
    print(len(yelp_pizza))
    print()
    print("Ratings distribution:")
    print(yelp_pizza.groupby('stars').size())
    print("Printing few rows of yelp_pizza")
    print(yelp_pizza.head())
    print(("Done!"))
    print()
    #Part 3: Creating our word embeddings
    print("Creating out word embeddings...")
    print("Loading GloVe embeddings...")
    word2vec_model_path = "/Volumes/Elements/December7th/GoogleNews-vectors-negative300.bin.gz"
    print("Creating embedding index")
    embedding_index = create_embedding_index(word2vec_model_path)

    vocabulary_size = 50000
    maxlen = 60
    num_dimensions = 300
    print("Creating sequence from text reviews")
    data, tokenizer = create_padded_sequence(vocabulary_size, maxlen, yelp_pizza, 'review')
    print("Creating embedding matrix")
    embedding_matrix = create_embedding_matrix(vocabulary_size,
                                               num_dimensions,
                                               tokenizer,
                                               embedding_index)

    print("Done!")
    print()
    #Part 4: Train test split, and one hot encoding the labels
    print("Creating one hot encoded labels")
    encoder = LabelBinarizer()
    one_hot_label = encoder.fit_transform(np.array(yelp_pizza[['stars']]))
    print("Splitting sequences into train/test")
    x_train, x_test, y_train, y_test = train_test_split(data, one_hot_label,
                                                        random_state=42,
                                                        stratify=one_hot_label,
                                                        test_size=0.1)

    print("Fitting Keras Model")
    #Part 5: Fitting our Keras models
    data_to_fit = {'x_train': x_train,
                   'x_test': x_test,
                   'y_train': y_train,
                   'y_test': y_test}

    model_ten_units = fit_lstms(num_units=10,
                                embedding_weights=[embedding_matrix],
                                num_epochs=100,
                                fit_data=data_to_fit)

    model_fifty_units = fit_lstms(num_units=50,
                                  embedding_weights=[embedding_matrix],
                                  num_epochs=100,
                                  fit_data=data_to_fit)

    model_hundred_units = fit_lstms(num_units=100,
                                    embedding_weights=[embedding_matrix],
                                    num_epochs=100,
                                    fit_data=data_to_fit)

    print(model_ten_units)
    print(model_fifty_units)
    print(model_hundred_units)

if __name__ == '__main__':
    main()
