import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def news_cla():
    fake = pd.read_csv('data\Fake.csv', encoding='latin1')
    real = pd.read_csv('data\True.csv', encoding='latin1')

    real_backup = real
    fake_backup = fake

    real['tf'] = 1
    fake['tf'] = 0

    real.drop(['text', 'subject', 'date'], axis=1, inplace=True)
    fake.drop(['text', 'subject', 'date'], axis=1, inplace=True)

    data = pd.concat([real, fake])
    data.groupby('tf').size()

    X_data = data['title'].values
    y_data = data['tf'].values

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_data) # 진짜 뉴스  행을 가진 X의 각 행에 토큰화를 수행
    sequences = tokenizer.texts_to_sequences(X_data)

    word_to_index = tokenizer.word_index

    threshold = 2
    total_cnt = len(word_to_index) # 단어의 수
    rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
    total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
    rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

    # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
    for key, value in tokenizer.word_counts.items():
        total_freq = total_freq + value

        # 단어의 등장 빈도수가 threshold보다 작으면
        if(value < threshold):
            rare_cnt = rare_cnt + 1
            rare_freq = rare_freq + value

    vocab_size = len(word_to_index)+1

    