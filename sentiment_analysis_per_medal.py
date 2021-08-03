# https://huggingface.co/monologg/koelectra-base-finetuned-nsmc
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained('monologg/koelectra-base-finetuned-nsmc')
model = AutoModelForSequenceClassification.from_pretrained('monologg/koelectra-base-finetuned-nsmc')
model = model.to(device)

filenames = ['./cleaned_data/'+filename for filename in os.listdir('./cleaned_data')]
medal_dict = {'gold':['금메달', '金'], 'silver':['은메달', '銀'], 'bronze': ['동메달', '銅']}

for medal in medal_dict:
    for filename in filenames:
        df = pd.read_csv(filename)

        medal_sents = []
        for news in df['news'].tolist():
            for sent in news.split('.'):
                if any(x in sent for x in medal_dict[medal]):
                    medal_sents.append(sent)

        preds = []
        # CUDA OOM 방지 위해 한 문장씩
        for sent in medal_sents:
            inputs = tokenizer(sent, return_tensors='pt', truncation=True).to(device)
            # (batch size, 2)
            logits = model(**inputs).logits
            score = torch.softmax(logits, -1)[-1][-1].item()
            preds.append(score)

        new_filename = './' + medal + '/' + filename.split('/')[2][:4] + '_' + medal +'_sentiment.csv'
        df = pd.DataFrame(columns=['sentence', 'sentiment'])
        df['sentence'] = medal_sents
        df['sentiment'] = preds
        df.to_csv(new_filename, index=False, encoding='utf-8')