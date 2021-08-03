import pandas as pd
import numpy as np
import os

filenames = [filename for filename in os.listdir() if filename.startswith('Sport_')]
words = ['메달','金','銀','銅','승리','진출','패배','탈락','연패','연승','완승','완패','역전']

def filter(news):
    if '올림픽' not in news:
        return np.nan
    if not any(word in news for word in words):
        return np.nan
    return news


for filename in filenames:
    df = pd.read_csv(filename, names=['datetime', 'category', 'ilbo', 'title', 'content', 'url'])
    df['news'] = df.apply(lambda x: str(x[3]) + '. ' + str(x[4]), axis=1)
    df = df[['datetime', 'news']]
    df['news'] = df['news'].apply(lambda x:filter(x))
    df = df.dropna()
    new_filename = '_'.join(filename.split('_')[2:])
    df.to_csv(new_filename, index=False, encoding='utf-8')

