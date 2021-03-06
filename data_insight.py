# -*- coding: utf-8 -*-
"""

根据观察结果, 对abstract 75%的文本长度少于53, 根据在mysql的统计结果, 除掉所有大于70的
对content, 去掉800单词以上的
"""
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

print(len(document[0]))
document_lengths = pd.Series([len(x.split()) for x in document])
summary_lengths = pd.Series([len(x.split()) for x in summary])
# 每篇文章长度

print("describe document: ", document_lengths.describe())
print("#######################################")
print("describe summary: ", summary_lengths.describe())
print("#######################################")