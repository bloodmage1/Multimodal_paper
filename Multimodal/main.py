import re
import os

import pandas as pd
import numpy as np
import librosa

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold

import torch
from transformers import PreTrainedTokenizerFast, PreTrainedModel
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer

from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from konlpy.tag import Okt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GRU, Dense
from tensorflow.keras.models import *
from keras.layers import *
from sklearn.utils.class_weight import compute_class_weight

from Function.preprocessing import *

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv('data.csv')
data = data[data["time"] < 30].reset_index(drop = True)
# data = data[data["sex"] == "F"].reset_index(drop = True) # 여자일 때
# data = data[data["sex"] == "M"].reset_index(drop = True) # 남자일 때

mean_Y = []

for i in tqdm(range(len(data))):
    wav ,sr = librosa.load(data["path"][i], sr = 22500)
    real_wav = signal.wiener(wav)  
    
    L = len(real_wav)
    n_fft = 76800 * 2
    N = int(n_fft/2)
    
    fft_data = np.fft.fft(real_wav, n = n_fft)
    Y =2*np.abs(fft_data[0:N])/(L/2)

    mean_Y.append(np.array(valid_mean_per_100(Y)))
    
tokenizer = Okt() #형태소 분석기 
token_list = []
korean_stopwords = ['을', '를', '이', '가', '은', '는','에'] #불용어 설정

data['text'] = data['text'].apply(lambda x: remove_stop(x, korean_stopwords))
data['evaluation'] = data['evaluation'].apply(lambda x: x.split(";")[0])

# Extract MFCC features
mfccs = []
max_length = 0
for path in tqdm(data['path']):
    audio, sample_rate = librosa.load(path, sr=22500)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=128)
    mfccs.append(mfcc)
    if len(mfcc[0]) > max_length:
        max_length = len(mfcc[0])

# Pad or truncate the MFCCs to the desired length
avg_length = int(np.mean([len(mfcc[0]) for mfcc in mfccs]))
for i in tqdm(range(len(mfccs))):
    mfcc_len = len(mfccs[i][0])
    if mfcc_len < avg_length:
        pad_width = avg_length - mfcc_len
        mfccs[i] = np.pad(mfccs[i], pad_width=((0,0), (0,pad_width)), mode='constant')
    elif mfcc_len > avg_length:
        mfccs[i] = mfccs[i][:, :avg_length]

# 80%는 trainset,  20%는 testset
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 문장(Text)
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(train_data['text'])

X_train_text = tokenizer.texts_to_sequences(train_data['text'])
X_test_text = tokenizer.texts_to_sequences(test_data['text'])

max_len = 128
X_train_text = pad_sequences(X_train_text, maxlen=max_len)
X_test_text = pad_sequences(X_test_text, maxlen=max_len)

# 음성 FFT(Audio)
train_data_audio = np.array(mean_Y)[:train_size]
test_data_audio = np.array(mean_Y)[train_size:]

# 음성 MFCC(Audio)
train_data_mfcc = np.array(mfccs)[:train_size]
test_data_mfcc = np.array(mfccs)[train_size:]
    
train_data_mfcc = np.array(mfccs)[:train_size]
test_data_mfcc = np.array(mfccs)[train_size:]

train_data_mfcc = np.transpose(train_data_mfcc, (0, 2, 1))
test_data_mfcc = np.transpose(test_data_mfcc, (0, 2, 1))

# 감정(Emotion)
y_train = train_data['evaluation'].tolist()
y_test = test_data['evaluation'].tolist()

ohe = OneHotEncoder(sparse_output=False)
y_train = ohe.fit_transform(np.array(y_train).reshape(-1, 1))
y_test = ohe.fit_transform(np.array(y_test).reshape(-1, 1))


### 1 ###

text_inputs = Input(shape=(max_len,))
audio_inputs = Input(shape=(768,))

# Text
x = Embedding(10000, 128, input_length=max_len)(text_inputs)
x = GlobalAveragePooling1D()(x)
text_mcb = tf.signal.fft(tf.cast(x, dtype=tf.complex64))

# Audio
y = Dense(128, activation='relu')(audio_inputs)
audio_mcb = tf.signal.fft(tf.cast(y, dtype=tf.complex64))

result = tf.math.real(tf.signal.ifft(text_mcb * audio_mcb))

outputs = Dense(7, activation='softmax')(result)
model = Model(inputs=[text_inputs, audio_inputs,], outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([X_train_text, train_data_audio], y_train, 
          epochs=30, batch_size=64, 
          validation_data=([X_test_text, test_data_audio], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_text, test_data_audio], y_test, 
                                verbose=0)

# Predict the test set results
y_pred = model.predict([X_test_text, test_data_audio])
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
print('Accuracy: %f' % (accuracy*100))


### 2 ###

text_inputs = Input(shape=(max_len,))
mfcc_inputs = Input(shape=(np.array(mfccs).shape[2],128))

# Text
x = Embedding(10000, 128, input_length=max_len)(text_inputs)
x = GlobalAveragePooling1D()(x)
text_mcb = tf.signal.fft(tf.cast(x, dtype=tf.complex64))

z = GlobalAveragePooling1D()(mfcc_inputs)
z = Dense(128, activation='relu')(z)
mfcc_mcb = tf.signal.fft(tf.cast(z, dtype=tf.complex64))

result = tf.math.real(tf.signal.ifft(text_mcb * mfcc_mcb))

outputs = Dense(7, activation='softmax')(result)
model = Model(inputs=[text_inputs, mfcc_inputs], outputs=outputs)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit([X_train_text,train_data_mfcc], y_train, 
          epochs=20, batch_size=64, 
          validation_data=([X_test_text, test_data_mfcc], y_test))

# Evaluate the model
loss, accuracy = model.evaluate([X_test_text, test_data_mfcc], y_test, 
                                verbose=0)

import numpy as np

# Predict the test set results
y_pred = model.predict([X_test_text, test_data_mfcc])

y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
print('Accuracy: %f' % (accuracy*100))