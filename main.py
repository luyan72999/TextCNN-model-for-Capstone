import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
import jieba
from tensorflow.keras.preprocessing.text import Tokenizer
from text_cnn import TextCNN
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import keras

# training params
max_features = 50000
maxlen = 30
batch_size = 1
embedding_dims = 50
epochs = 3

# Load and preprocess dataset
column_names = ['Characters', 'Pinyin', 'Meaning', 'HSK average', 'Custom Ratio']
df = pd.read_csv('./sentences.tsv', sep='\t', skiprows=1, names=column_names)
texts = df['Characters'].tolist()
labels = df['HSK average'].tolist()

# manually encode labels to 3 categories
# df['label'] = df['label'].apply(lambda x: 1 if x <= 3 else (2 if x <= 5 else 3)) # Map HSK average values to 1, 2, 3
y_data = []
for label in labels:
    if label <= 3:
        y_data.append(0)
    elif label <= 5:
        y_data.append(1)
    else:
        y_data.append(2)

# tokenize sentences
tokens = []
for sentence in texts:
    # word segmentation using jieba
    seg_list = jieba.lcut(sentence)
    
    # Remove punctuation
    seg_list_clean = [token for token in seg_list if token.strip() and not re.match(r"[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）]+", token)]
    
    tokens.append(seg_list_clean)

# use Tokenizer for vectorization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)

# convert texts to sequences of integers
sequences = tokenizer.texts_to_sequences(tokens)

# pad sequences to the same length
x_data = sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')

# split train and test data 80%: 20%
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
# convert data type to fit into model
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = TextCNN(maxlen, max_features, embedding_dims)
#  the model learns the optimal weights and biases that minimize the difference between predicted and actual labels. 
# This is typically done using gradient-based optimization algorithms, such as Adam.
model.compile('adam', 'sparse_categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy(), keras.metrics.SparseCategoricalCrossentropy()])

print('Training...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

print('Testing...')
result = model.predict(x_test)
'''
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')
print(f'Test Accuracy: {accuracy}')
This can be omitted as val_accuracy already indicates performance on test data
'''
