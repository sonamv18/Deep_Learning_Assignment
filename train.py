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

train = pd.read_csv('./train_data.csv')
train, val = train_test_split(train, test_size=0.25, random_state=42)

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

train_text = preprocess(train['transcription'])
val_text = preprocess(val['transcription'])

def encode(train_label, val_label):
  le = preprocessing.LabelEncoder()
  le.fit(list(train_label.values))
  y_train_label = le.transform(list(train_label.values))
  y_val_label = le.transform(list(val_label.values))
  return tf.keras.utils.to_categorical(y_train_label), y_train_label, tf.keras.utils.to_categorical(y_val_label), y_val_label

y_train_action_b, y_train_action, y_val_action_b, y_val_action = encode(train['action'], val['action'])
y_train_object_b, y_train_object, y_val_object_b, y_val_object = encode(train['object'], val['object'])
y_train_location_b, y_train_location, y_val_location_b, y_val_location = encode(train['location'], val['location'])

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_text)
train_sequences = tokenizer.texts_to_sequences(train_text)
val_sequences = tokenizer.texts_to_sequences(val_text)
X_train = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=10, padding='pre', truncating='pre')
X_val = tf.keras.preprocessing.sequence.pad_sequences(val_sequences, maxlen=10, padding='pre', truncating='pre')

tokenizer_json = tokenizer.to_json()
with io.open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(json.dumps(tokenizer_json, ensure_ascii=False))

e_model = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
word_index = tokenizer.word_index
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    try:    
      embedding_vector = e_model[word]
      if embedding_vector is not None:
          embedding_matrix[i] = embedding_vector
    except:
      embedding_matrix[i] = np.zeros((300,))
      print(1)

mirrored_strategy = tf.distribute.MirroredStrategy()
mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

MAX_SEQUENCE_LENGTH = 10
tf.keras.backend.clear_session()

with mirrored_strategy.scope():
  inputs = tf.keras.layers.Input(shape=(MAX_SEQUENCE_LENGTH,), name='Tokens')
  embedding_layer = tf.keras.layers.Embedding(len(word_index)+1, 300, weights=[embedding_matrix], trainable=False)(inputs)
  LSTM_Layer1 = tf.keras.layers.LSTM(128)(embedding_layer)

  output1 = tf.keras.layers.Dense(6, activation='softmax')(LSTM_Layer1)
  output2 = tf.keras.layers.Dense(14, activation='softmax')(LSTM_Layer1)
  output3 = tf.keras.layers.Dense(4, activation='softmax')(LSTM_Layer1)

  model = tf.keras.Model(inputs=inputs, outputs=[output1, output2, output3])
  callback_save = tf.keras.callbacks.ModelCheckpoint("model_save.h5",monitor="val_loss", save_best_only=True)

  def scheduler(epoch, lr):
    if epoch < 10:
      return lr
    else:
      return lr * tf.math.exp(-0.1)

  learningRate_Schedular = tf.keras.callbacks.LearningRateScheduler(scheduler)

  LRpt = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=0)
  log_dir="logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True,write_grads=True)

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
callbacks_list = [learningRate_Schedular, LRpt, tensorboard_callback, callback_save]

history=model.fit(X_train, [y_train_action_b, y_train_object_b, y_train_location_b], batch_size=100, epochs=50, 
                  validation_data=(X_val, [y_val_action_b, y_val_object_b, y_val_location_b]), callbacks=callbacks_list)

