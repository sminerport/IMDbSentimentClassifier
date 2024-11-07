import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show errors

# Now import all other libraries
import pandas as pd
import numpy as np
import urllib.request
import zipfile
import glob
import sys
import re

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

current_version = re.match(r"(\d+\.\d+\.\d+)", sys.version).group(0)

version_file = "python_version.txt"

if os.path.exists(version_file):
    with open(version_file, "r") as f:
        saved_version = f.read().strip()

    if saved_version != f"Python version: {current_version}":
        print("Python version has changed, updating the version file.")
        with open(version_file, "w") as f:
            f.write(f"Python version: {current_version}\n")
    else:
        print("Python version is the same, no update needed.")
else:
    print("Version file not found, creating a new one.")
    with open(version_file, "w") as f:
        f.write(f"Python version: {current_version}\n")

# Create the models directory if it doesn't exist
models_folder = 'models'
os.makedirs(models_folder, exist_ok=True)
model_path = os.path.join(models_folder, 'best_model.keras')

# File paths
TRAIN_CSV = 'Train.csv'
VALID_CSV = 'Valid.csv'
TEST_CSV = 'Test.csv'
training = os.path.join('data', TRAIN_CSV)
validation = os.path.join('data', VALID_CSV)
testing = os.path.join('data', TEST_CSV)

# Load data
train = pd.read_csv(training)
valid = pd.read_csv(validation)
test = pd.read_csv(testing)

#train-test split
x_tr, y_tr = train['text'].values, train['label'].values
x_val, y_val = valid['text'].values, valid['label'].values
x_text, y_test = test['text'].values, test['label'].values

#Tokenize the sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(list(x_tr))
x_tr_seq = tokenizer.texts_to_sequences(x_tr)
x_val_seq = tokenizer.texts_to_sequences(x_val)
x_test_seq = tokenizer.texts_to_sequences(x_text)
x_tr_seq = pad_sequences(x_tr_seq, maxlen=100)
x_val_seq = pad_sequences(x_val_seq, maxlen=100)
x_test_seq = pad_sequences(x_test_seq, maxlen=100)

# Vocabulary size
size_of_vocabulary = len(tokenizer.word_index) + 1 
print(f"Size of vocab: {size_of_vocabulary}")

# Folder to store GloVe embeddings
glove_folder = 'glove'
os.makedirs(glove_folder, exist_ok=True)
glove_path = os.path.join(glove_folder, 'glove.6B.100d.txt')
download_path = os.path.join(glove_folder, "glove.6B.zip")

if not os.path.exists(glove_path):
    print("Downloading GloVe embeddings...")
    urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", download_path)
    with zipfile.ZipFile(download_path, 'r') as zip_ref:
        zip_ref.extractall(glove_folder)
    print("GloVe embeddings downloaded and extracted.")

    # Remove all GloVe files except 'glove.6B100d.txt'
    for file in glob.glob(os.path.join(glove_folder, 'glove.6B.*')):
        if 'glove.6B.100d.txt' not in file:
            os.remove(file)

# Load GloVe embeddings into an embedding matrix
embedding_dim = 100
embedding_index = {}
with open(glove_path, 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vectors

embedding_matrix = np.zeros((size_of_vocabulary, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Model architecture
model = Sequential()

# Embedding layer with pretrained embeddings
model.add(Embedding(size_of_vocabulary, embedding_dim, weights=[embedding_matrix],
                    trainable=False))

# Bidirectional LSTM with Batch Normalization and Dropout
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)))
model.add(BatchNormalization())
model.add(GlobalMaxPooling1D())

# Dense layer with L2 regularization and dropout
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

# Callbacks
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint(model_path, monitor='val_acc', mode='max',
                     save_best_only=True, verbose=1)

# Suppress model summary warnings
print(model.summary())

# Train the model only if best_model.keras doesn't exist
if not os.path.exists(model_path):
    history = model.fit(
        np.array(x_tr_seq), np.array(y_tr),
        batch_size=128,
        epochs=10,
        validation_data=(np.array(x_val_seq), np.array(y_val)),
        verbose=1,
        callbacks=[es, mc]
    )

# Load the best model and evaluate
model = load_model(model_path)
_, val_acc = model.evaluate(np.array(x_val_seq), np.array(y_val), batch_size=128)
print(f"Validation Accuracy: {val_acc}")

# Evaluate on the test set
_, test_acc = model.evaluate(np.array(x_test_seq), np.array(y_test), batch_size=128)
print(f"Test Accuracy: {test_acc}")

cleanup = True  # Set to True if you want to delete files after each run

if cleanup:
    if os.path.exists(model_path):
        os.remove(model_path)
        print("Deleted best_model.keras to save space.")
    
    if os.path.exists(glove_path):
        os.remove(glove_path)
        print("Deleted glove.6B.100d.txt to save space.")