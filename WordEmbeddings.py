# import libraries
import pandas as pd
import numpy as np
import os

TRAIN_CSV = 'Train.csv'
VALID_CSV = 'Valid.csv'
training = os.path.join('data', TRAIN_CSV)
validation = os.path.join('data', VALID_CSV)
# reading csv files
train = pd.read_csv(training)
valid = pd.read_csv(validation)

#train_test split
x_tr, y_tr = train['text'].values, train['label'].values
x_val, y_val = valid['text'].values, valid['label'].values

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#Tokenize the sentences
tokenizer = Tokenizer()
#preparing vocabulary
tokenizer.fit_on_texts(list(x_tr))

# converting text into integer sequences
x_tr_seq = tokenizer.texts_to_sequences(x_tr)
x_val_seq = tokenizer.texts_to_sequences(x_val)

# padding to prepare sequences of same length
x_tr_seq = pad_sequences(x_tr_seq, maxlen=100)
x_val_seq = pad_sequences(x_val_seq, maxleng=100)

size_of_vocabulary=len(tokenizer.word_index) + 1 #+1 for padding
print(size_of_vocabulary)

# build two different NLP models of the same architecture. The first learns embeddings from scratch
# the second uses pretrained word embeddings

