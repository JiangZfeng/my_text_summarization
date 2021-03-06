# abstractive_summarizer
# 使用transformer 模型进行文本摘要生成, 
参考tensorflow 官方的transformer模型


origin_model.py 为原作者模型与相关数据
main_model_transformer_saveweight.py 为经过修改, 使用自己数据, 并修改为能够保存训练的模型
predict_transformer.py 为导入训练完成的模型与配置的文件

loss: SparseCategoricalCrossentropy
metric: accuracy/ rouge_l

数据存放于./data
数据爬取子环球时报, 截去过长和过短的部分

Abstractive Text Summarization using Transformer

- Implementation of the state of the art Transformer Model from "Attention is all you need", Vaswani et. al.
  https://arxiv.org/abs/1706.03762

- Inshorts Dataset: https://www.kaggle.com/shashichander009/inshorts-news-data


Blog Links:

Part-I: https://towardsdatascience.com/transformers-explained-65454c0f3fa7

Part-II: https://medium.com/swlh/abstractive-text-summarization-using-transformers-3e774cc42453


License: [Apache License 2.0](https://github.com/rojagtap/abstractive_summarizer/blob/master/LICENSE)
