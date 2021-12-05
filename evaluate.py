

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re
from sklearn import preprocessing
from gensim.models import KeyedVectors
import io
import json
import datetime 

test = pd.read_csv('./valid_data.csv')

def decontracted(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess(raw):
  preprocessed = []
  # tqdm is for printing the status bar
  for sentance in tqdm(raw):
    sentance = re.sub(r"http\S+", "", sentance)  #removing html tags
    sentance = decontracted(sentance) #decontrast
    sentance = ' '.join(e.lower() for e in sentance.split()) #lowering
    sentance = re.sub('[^a-zA-Z]', ' ', sentance) #removing puncuation and numericals
    sentance = re.sub(r'\s+', ' ', sentance) #remving wide spaces
    preprocessed.append(sentance.strip())
  return preprocessed 

test_text = preprocess(test['transcription'])

def encode(test_label):
  le = preprocessing.LabelEncoder()
  le.fit(list(test_label.values))
  y_test_label = le.transform(list(test_label.values))
  return tf.keras.utils.to_categorical(y_test_label), y_test_label

y_test_action_b, y_test_action = encode(test['action'])
y_test_object_b, y_test_object = encode(test['object'])
y_test_location_b, y_test_location = encode(test['location'])


with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(data)

test_sequences = tokenizer.texts_to_sequences(test_text)
X_test = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=10, padding='pre', truncating='pre')

model = tf.keras.models.load_model('./model_save.h5')

result = model.evaluate(X_test, [y_test_action_b, y_test_object_b, y_test_location_b])

print(result)