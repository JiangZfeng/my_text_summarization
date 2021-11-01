#from main_model_transformer_saveweights import create_masks
import tensorflow as tf
import pandas as pd
import numpy as np


num_layers = 4      # 编码器 里的 编码层 层数
d_model = 128       # 词嵌入维度
dff = 512           # 前馈网络中的节点数
num_heads = 8       # 多头数
EPOCHS = 20
# buffer
BUFFER_SIZE = 2000         # shuffle
BATCH_SIZE = 64


test_text = []
with open('data/test.txt','r') as f:
    test_text = f.read()

new_model = tf.keras.models.load_model('saved_model_dir')
print(new_model.summary())

def create_masks(inp, tar):  # input  target  根据每次迭代的input 和 target 来生成mask
    enc_padding_mask = create_padding_mask(inp)  # 编码和解码的mask
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # 对未生成单词的mask
    dec_target_padding_mask = create_padding_mask(tar)  # 不知道
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 取最大?怎么比较大小

    return enc_padding_mask, combined_mask, dec_padding_mask  # 三个

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)    # 判断是否与0相等
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

def evaluate(input_document):
    # 序列化输入,生成词索引 填充
    input_document = document_tokenizer.texts_to_sequences([input_document])
    input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=encoder_maxlen,
                                                                   padding='post', truncating='post')

    # expand 拓展一个维度, 0 代表将第0维拓展
    encoder_input = tf.expand_dims(input_document[0], 0)  # 输入第0个input_document?
    # 初始输入为开头标记
    decoder_input = [summary_tokenizer.word_index["<go>"]]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(decoder_maxlen):
        # 循环生成单词, 直到达到最大长度
        # mask, 动态变化

        # 调用之前创建的transformer, 将training置为false
        predictions, attention_weights = new_model(
            [encoder_input,output],
            training=False,
        )

        # 根据argmax 找到对应最大的 词索引
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if predicted_id == summary_tokenizer.word_index["<stop>"]:
            # it is down
            print("predict over")
            return tf.squeeze(output, axis=0), attention_weights
        # 词索引序列
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def summarize(input_document):
    # not considering attention weights for now, can be used to plot attention heatmaps in the future
    # 没有使用attention weight
    summarized = evaluate(input_document=input_document)[0].numpy()
    print("document:  ",input_document)
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document
    # 将token 变回文本




news = pd.read_excel("data/news_globaltimes.xlsx")

# 清理数据
# news.drop(['news_id', 'url', 'pub_time', 'article_level_one', 'article_level_two','title'], axis=1, inplace=True)
news = news.dropna(axis='index',how='any')
# 分开
document = news['content']
summary = news['abstract']

summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'

document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)

document_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)

encoder_vocab_size = len(document_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1

encoder_maxlen = 1500
decoder_maxlen = 300

ret = summarize(test_text)
print(ret)

