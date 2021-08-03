import pandas as pd
from gensim.models import Word2Vec
from konlpy.tag import Mecab
import logging
import os

filenames = [filename for filename in os.listdir() if filename.startswith('20') and filename.endswith('.csv') and 'sentiment' not in filename]
mecab = Mecab()

def pick_morphs(news):
    return mecab.morphs(news)

for filename in filenames:
    df = pd.read_csv(filename)
    df['tokens'] = df['news'].apply(lambda x: pick_morphs(x))

    # word2vec 모델 학습에 로그 찍기
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

    # 모델 학습하기
    # window 10, 5 비교
    model = Word2Vec(df['tokens'], sg=1, size=100, window=2, min_count=2, workers=4)

    # 모델 저장하기
    model_name = filename.split('_')[0] + '_window2'
    model.save(model_name)

# 저장한 모델 불러오기
# model = Word2Vec.load(model_name)
# model.wv.most_similar('금메달')