
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import re
import pickle
# import tensorflow_text as text
###########################################
# 超参数设置
# hyper-params
# 层数,  xxx, xxx, 注意力机制的heads, 时代数
num_layers = 4      # 编码器 里的 编码层 层数
d_model = 128       # 词嵌入维度
dff = 512           # 前馈网络中的节点数
num_heads = 8       # 多头数
EPOCHS = 20
# 3000 samples   too less to training
# epoch 5   all outputs  are   <the>
# epoch 10  outputs  art     <the>  <and> <us> <of>
# epoch 20  0-15 words is ok, but rest words is repeated

# buffer
BUFFER_SIZE = 2000         # shuffle e samples per 1000
BATCH_SIZE = 32


#####################################################
# 导入数据与预处理
# 导入数据
news = pd.read_excel("data/news_lessthan_500words.xlsx")
# 清理数据
# news.drop(['news_id', 'url', 'pub_time', 'article_level_one', 'article_level_two','title'], axis=1, inplace=True)
# 查看数据
print(news.head())
print(len(news))
print(news[news['content'].isnull()])
news = news.dropna(axis='index',how='any')      # remove the lines that has NaN
print(len(news))

document = news['content']
summary = news['abstract']



# for decoder sequence     #做标记, 标记开始与结束
summary = summary.apply(lambda x: '<go> ' + x + ' <stop>')
print(document.describe())
print(summary.describe())


#########################################################
# 生成词索引

filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
oov_token = '<unk>'
# 标点符号和未知单词的 处理


# 包含对文本过滤, 小写化, 分词, 生成词索引
document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
# fit
document_tokenizer.fit_on_texts(document)
summary_tokenizer.fit_on_texts(summary)
# 将文本索引化
inputs = document_tokenizer.texts_to_sequences(document)
targets = summary_tokenizer.texts_to_sequences(summary)
# 序列化, 变成词列表
# 已经变成了token列表


# vocab_size  两个词典的大小
encoder_vocab_size = len(document_tokenizer.word_index) + 1
decoder_vocab_size = len(summary_tokenizer.word_index) + 1


##############################################################
# 对数据做一个简单的观察
#
# document_lengths = pd.Series([len(x) for x in document])
# summary_lengths = pd.Series([len(x) for x in summary])
# # 每篇文章长度
#
# print("describe document: ", document_lengths.describe())
# print("describe summary: ", summary_lengths.describe())

# 根据前面的统计, 设置合适的长度
# he mean length of inputs is 3000
# the mean length of outputs is 300
# but the memory is exhausted, so change the maxlen
# 发现本模型对于太长的文本序列无能为力, 生成的结果牛头不对马嘴, 改成 maxlen=500
encoder_maxlen = 500
decoder_maxlen = 70


##################################################################
# 填充与截断
inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=encoder_maxlen, padding='post', truncating='post')
targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=decoder_maxlen, padding='post', truncating='post')# 用post, 向后面填充


inputs = tf.cast(inputs, dtype=tf.int32)    # 转化为张量
targets = tf.cast(targets, dtype=tf.int32)
# 生成batch, 打乱数据
dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)



#################################################################
# 进行位置编码
# 在单词之间添加位置编码, 能够使得具有相似性的单词在d维空间中更靠近
def get_angles(position, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return position * angle_rates
# 计算位置编码向量
def positional_encoding(position, d_model):
    # 位置编码  d_model长
    # 获得输入单词间的位置关系
    #newaxis 给数组加一维, 改变方向?
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    # 正弦 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # 余弦 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)



####################################################################################
### Masking
# 两个mask, 一个用于遮蔽填充字符,
# 一个用于遮蔽 未来要生成的单词, 以免影响当前单词预测
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)    # 判断是否与0相等
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

#########################################################################
### Building the Model

# Scaled Dot Product 拓展点乘  计算注意力的核心
def scaled_dot_product_attention(q, k, v, mask):
    # 即输入q, 根据q 与 k 的相似性找到 最匹配的 v
    # q,k 进行 矩阵点乘, 得到一个相关性分数
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    # 得到缩放系数dk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    # 除以 根号dk(即向量长度) 缩放, 保持梯度的稳定
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # mask, 对尚未考虑的单词遮蔽
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # 用mask遮蔽, 对注意力做softmax, 放大差距
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    # output 是注意力权重矩阵和 矩阵v的乘法
    # softmax结果与v 点乘, 即对 v 中的单词加权处理
    output = tf.matmul(attention_weights, v)
    # 返回注意力 和 注意力权重
    return output, attention_weights



#############################################################################
# Multi-Headed Attention
# 多头注意力层   可以并行计算
class MultiHeadAttention(tf.keras.layers.Layer):
    """多头注意力机制"""
    # muti-head层
    # 有多个dense层组成
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0  # 要求能够整除

        self.depth = d_model // self.num_heads  # 所谓深度, 一个头的数据的维度

        self.wq = tf.keras.layers.Dense(d_model)    # 全连接层
        self.wk = tf.keras.layers.Dense(d_model)    # 实际是对应于q,k,v 的权值矩阵
        self.wv = tf.keras.layers.Dense(d_model)    # 对q,k,v 投影, 可以学习

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):  # 将输入的向量reshape为多头
        # batch = 64 head = 8 depth = 16
        # 每个序列长度length = -1 ???
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # 对列交换顺序? 为什么是0 2 1 3

    def call(self, v, k, q, mask):
        # 输入的v,k,q 形状相同
        # v,k 是相同的
        batch_size = tf.shape(q)[0]

        # 还是dmodel 的词向量
        # q,k,v 由词嵌入x 分别乘三个对应的权值矩阵得到
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)  # 经过reshape 变成矩阵
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # 进行scaled product
        # 进行矩阵间的乘法, 分解开就是计算每个词向量 与 其他词向量 的关系, 类似用点乘计算相关性
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # 转置, 排列
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # 将多头分割的向量 合成
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)

        return output, attention_weights


################################################################
# Feed Forward Network 进行正则化 与 生成词嵌入
# d_model 为词嵌入的维度
def point_wise_feed_forward_network(d_model, dff):
    # 逐点前馈网络
    # 几乎每一个后都加上, 为了生成词向量?
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

####################################################################
# Fundamental Unit of Transformer encoder
# 将前面介绍的 多头注意力层, 前馈层, 连接组合为编码层

class EncoderLayer(tf.keras.layers.Layer):
    # transformer 编码器的基本单元
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)           # mutil-head attention
        self.ffn = point_wise_feed_forward_network(d_model, dff)  # 逐点前馈网络层
        # 归一化
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)  # 进行层标准化
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # 正则化
        self.dropout1 = tf.keras.layers.Dropout(rate)       # 正则化
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):  # 调用encoder层
        # x为词嵌入序列
        # mha层的输出
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # 类似残差连接
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        # 残差连接 正则化
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


##############################################################################
#### Fundamental Unit of Transformer decoder
class DecoderLayer(tf.keras.layers.Layer):
    # 解码层
    # 一个解码层包含一个mutil-head, 一个mask的muti-head, 一个全连接层
    # 每层后面都有标准化和正则化层
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        # dmodel为的向量,
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # 第一个是接收 目标输出的层
        # 自身各个单词之间的关系, 用lookahead mask
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)    # q,k,v都是x
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        # 第二个是 接收编码器输入和目标输出的 层
        # 编码层输出 与 目标输出的关系, 用padding mask
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, padding_mask)  # q,k,v
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2   # 两个attention



###############################################################################
# Encoder consisting of multiple EncoderLayer(s)
# 多个编码层 又 组成一个编码器
class Encoder(tf.keras.layers.Layer):
    # 多层编码层 叠加成 一个编码器
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)  # 词嵌入层, d_model维向量
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)  # 位置编码函数

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]  # 四个编码层组成以编码器

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # 调用编码器
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)  # num 层叠加

        return x

#######################################################################
# Decoder consisting of multiple DecoderLayer(s)
class Decoder(tf.keras.layers.Layer):
    # 解码器
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers    # 解码器里面有num个 解码层
        # 进行词嵌入和位置编码
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]  # 同理,多个解码层
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]    # 得到序列长度
        attention_weights = {}      # 注意力权值

        x = self.embedding(x)       # 进行词嵌入
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            # 将多个解码层的attention 矩阵结合为一个总的attention词典
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        return x, attention_weights


########################################################################
# Finally, the Transformer
# transformer 是 输入, 输出, 编码器, 解码器的组合
class Transformer(tf.keras.Model):
    # 最终的transformer
    # 自定义模型
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1):
        super(Transformer, self).__init__()
        # 编码器
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        # 解码器
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        # 最后的全连接层, 计算得到每一步的输出
        self.final_layer = tf.keras.layers.Dense(target_vocab_size) # 节点数为目标词典大小


    def call(self, inputs, training ):
        # 每当调用Transformer实例时, 就是调用此函数
        # training 是否进行训练, 是否反向传播
        # 前向传播
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = create_masks(inp, tar)    # mask不断变化
        enc_output = self.encoder(inp, training, enc_padding_mask)  # 编码器的输出 编码器的mask

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)

        return final_output, attention_weights  # 得到最后的输出与所有的attention权重

    # write get_config funtion for loading model???


###############################################################################
# Training
# Adam optimizer with custom learning rate scheduling
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # adam优化器, 自定义学习率
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)      # 随步数调整?
        arg2 = step * (self.warmup_steps ** -1.5)

        # 学习率, 论文中给出的公式
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


###############################################################################
# 定义学习率
learning_rate = CustomSchedule(d_model) # 调用call, 初始化学习率
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
# 稀疏分类交叉熵

# lossfunction
def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))  # 判断real与0是否相等, 结果逻辑取反
    loss_ = loss_object(real, pred)                     # lossobject 为稀疏分类交叉熵

    mask = tf.cast(mask, dtype=loss_.dtype)             # mask也是交叉熵?
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)     # 求和???

# metrics
def accuracy_function(real,pred):
    # pred = tf.convert_to_tensor(pred,dtype=tf.int32)
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=2), dtype=tf.int32))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

#################################################################################
# 定义transformer
# 定义transformer结构
transformer = Transformer(
    num_layers,
    d_model,
    num_heads,
    dff,
    encoder_vocab_size,     # 词嵌入数计算的词典大小
    decoder_vocab_size,
    pe_input=encoder_vocab_size,
    pe_target=decoder_vocab_size,
)


# 定义transformer需要用的mask
def create_masks(inp, tar):  # input  target  根据每次迭代的input 和 target 来生成mask
    enc_padding_mask = create_padding_mask(inp)  # 编码和解码的mask
    dec_padding_mask = create_padding_mask(inp)

    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])  # 对未生成单词的mask
    dec_target_padding_mask = create_padding_mask(tar)  # 不知道
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)  # 取最大?怎么比较大小

    return enc_padding_mask, combined_mask, dec_padding_mask  # 三个


#################################################################################


##############################################################################
# 模型保存，注意：仅仅是多了一个save_format的参数而已
# 网络结构、权重、配置、优化器状态
# 注意：这里的'path_to_saved_model'不再是模型名称，仅仅是一个文件夹，模型会保存在这个文件夹之下
#transformer.save('path_to_saved_model', save_format='tf')

# 加载模型，通过指定存放模型的文件夹来加载
# new_model = tf.keras.models.load_model('path_to_saved_model')


###############################################################################
# Training steps
# 输入数据的形状 为了能够保存模型model.save()
train_step_signature = [
    tf.TensorSpec(shape=(None,None),dtype=tf.int32),
    tf.TensorSpec(shape=(None,None),dtype=tf.int32),
]
@tf.function(input_signature=train_step_signature)    # 高性能模式  静态图
def train_step(inp, tar):
    tar_inp = tar[:, :-1]       # 所有
    tar_real = tar[:, 1:]       # 忽略第一个开始的标志

    with tf.GradientTape() as tape:         # 梯度流 会监控所有可训练的变量
        predictions, _ = transformer(       # 没有用到权重 _
            [inp, tar_inp],                 # 输入与目标输出
            training=True,                  # True 表示进行训练

        )
        loss = loss_function(tar_real, predictions)     # 两者对比, 计算loss

    gradients = tape.gradient(loss, transformer.trainable_variables)    # 计算梯度, 所有能训练的变量
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))  # 优化器?
    # 将训练结果apply?

    train_loss(loss)        # 定义的损失函数object
    train_accuracy(accuracy_function(tar_real, predictions)) # tar: int32  pred: float32


#############################################################
# 训练过程
loss_list = []
accuracy_list = []
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()  # ???
    train_accuracy.reset_states()

    for (batch, (inp, tar)) in enumerate(dataset):
        # 生成数据batch
        # 输入batch个 文本和title, 开始训练
        train_step(inp, tar)


        if batch % 50 == 0:  #
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    # if (epoch + 1) % 2 == 0:        # 两代保存一次
    #     # transformer.save('path_to_saved_model', save_format='tf') ot work
    #     # cannot overwrite the save_model ???
    #     print('epoch {} '.format(epoch + 1))
        loss_list.append(train_loss.result())
        accuracy_list.append(train_accuracy.result())

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

transformer.save('saved_model_dir', save_format='tf')
np.save("results/loss_result_epochs_20", loss_list)
np.save("results/acc_result_epochs_20", accuracy_list)

#transformer.save('saved_model_dir/model.h5')


################################################################################
#### Predicting one word at a time at the decoder and appending it to the output; then taking the complete sequence as an input to the decoder and repeating until maxlen or stop keyword appears
# 预测

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
        predictions, attention_weights = transformer(
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
    summarized = np.expand_dims(summarized[1:], 0)  # not printing <go> token
    return summary_tokenizer.sequences_to_texts(summarized)[0]  # since there is just one translated document
    # 将token 变回文本


###############################################################################################
#
print("########################################3")
tmp = summarize(
    "US-based private equity firm General Atlantic is in talks to invest about \
    $850 million to $950 million in Reliance Industries' digital unit Jio \
    Platforms, the Bloomberg reported. Saudi Arabia's $320 billion sovereign \
    wealth fund is reportedly also exploring a potential investment in the \
    Mukesh Ambani-led company. The 'Public Investment Fund' is looking to \
    acquire a minority stake in Jio Platforms."
)
print(tmp)
print("########################################3")



































