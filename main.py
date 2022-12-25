import requests
import json
import numpy as np
import pandas as pd
import nltk 
from nltk.corpus import stopwords 
import numpy as np

df = pd.read_csv("Data.csv")

from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
sia = SIA()
results = []
for line in df['TITLE'] :
    score = sia.polarity_scores(line)
    score['headline'] = line
    results.append(score)

headlines_polarity = pd.DataFrame.from_records(results)

headlines_polarity['label'] = 0
headlines_polarity.loc[headlines_polarity['compound'] > 0.1, 'label'] = 1
headlines_polarity.loc[headlines_polarity['compound'] < -0.1, 'label'] = -1

headlines_polarity.head()

