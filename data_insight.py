# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

news = pd.read_excel("data/news_globaltimes.xlsx")

print(news.head())
print(len(news))

print("#######################################")
print(news[news['content'].isnull()])
print("#######################################")
print(news[news['abstract'].isnull()])

news = news.dropna(axis='index',how='any')


document = news['content']
summary = news['abstract']
print(document.describe())
print("#######################################")
print(summary.describe())
print("#######################################")


document_lengths = pd.Series([len(x) for x in document])
summary_lengths = pd.Series([len(x) for x in summary])
# 每篇文章长度

print("describe document: ", document_lengths.describe())
print("#######################################")
print("describe summary: ", summary_lengths.describe())
print("#######################################")