# 欢迎来到暴力代码的世界
# 本暴力代码是将图片当中的文字进行识别，思路是设定一个最长长度的限制，然后进行识别，识别结果为定长向量
# 本暴力代码的输入图片将直接归一化为256*256的格式
# 这里使用一个MAX_SIZE(最多的字符串个数)*MOST_CH(最多的)的one-hot向量来对结果进行显示

import numpy as np
import tensorflow as tf
import PicCreater
import PicReader
import Config
from tensorflow.contrib import rnn
from utils import *

# 将图片的宽高转换为固定的值
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

# batch_size
BATCH_SIZE = 64

# 彩色或灰度图像 现在是直接灰度
RGB = 1

# 字符串的最大长度
LONGEST_CH = 100
# 支持的字符数量
SUPPORT_CH = 10

# batch_size = tf.placeholder(tf.int32, name='batch_size')

X = tf.placeholder(tf.float32, shape = [None, IMAGE_HEIGHT, IMAGE_WIDTH], name='X')
Y = tf.placeholder(tf.float32, shape = [BATCH_SIZE, LONGEST_CH], name='Y')

keep_prob = tf.placeholder(tf.float32)

batch_x, batch_y, num_out = PicCreater.getNextBatchMem(64, IMAGE_WIDTH, IMAGE_HEIGHT)

num_predict = np.zeros(shape=[64])

train_phase = tf.placeholder(tf.bool)

def batch_norm(x, beta, gamma, phase_train, scope='bn', decay=0.9, eps=1e-5):
    '''
    进行序列化操作
    :param x: 
    :param beta: 
    :param gamma: 
    :param phase_train: 
    :param scope: 
    :param decay: 
    :param eps: 
    :return: 
    '''
    with tf.variable_scope(scope):
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(phase_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
    # mean,var = mean_var_with_update()
    normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, eps)
    return normed

# 卷积层
def conv_layer(batch_x):
    print('begin conv')

    print('batch_x', tf.shape(batch_x))
    x = tf.reshape(batch_x, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, RGB])
    print('reshape over')
    # 第一层卷积
    w_conv1 = tf.Variable(tf.random_normal([5, 5, RGB, 32]), name='w_conv1')
    b_conv1 = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32]), name='b_conv1')

    # 先卷积
    conv1 = tf.nn.conv2d(x, w_conv1, [1,1,1,1], padding='SAME')
    # 然后加上偏置项
    conv1 = tf.nn.bias_add(conv1, b_conv1)
    # 进行激活
    conv1 = tf.nn.relu(conv1)
    # bn一层
    conv1 = batch_norm(conv1, tf.constant(0.0, shape=[32]), tf.random_normal(shape=[32], mean=1.0, stddev=0.02),
                       train_phase, scope='bn_1')


    # 池化
    conv1 = tf.nn.max_pool(conv1, [1,2,2,1], [1,1,1,1], padding='SAME')

    return conv1

def connect_layer(input, Max_Len):
    '''
    全连接层, 将input转换为[batch_size, -1]，输出的w的值为[-1, class_num]
    :param input: 要进行全连接的张量
    :param class_num: 进行分类的类别数量
    :return: 分类后的向量集合,形式为[batch_size, class_num]
    '''

    print(tf.shape(input))

    batch_size_temp = tf.Variable(tf.shape(input)[0])

    input_X = tf.reshape(input, shape=[batch_size_temp, -1])

    # 定义全连接层的w和b
    connect_w = tf.Variable(tf.random_normal(shape=[256*256, Max_Len]))
    connect_b = tf.Variable(tf.zeros(shape=[64, Max_Len]))

    output = tf.nn.relu(tf.add(tf.matmul(input_X, connect_w), connect_b))

    # 直接把输入转换为
    return output

# lstm层
def lstm_layer(input, batch_size, hidden_size=256, layer_num=2):

    #hidden_size = time_step

    # 对输入数据进行塑型 暴力
    input_X = tf.reshape(input, [-1, 256, 256*32])
    # # 单独的LSTM层
    # lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    # # dropout
    # lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    # # 多层LSTM
    # mlstm_cell = rnn.MultiRNNCell([lstm_cell]*layer_num)

    def lstm_cell():
        lstm = rnn.BasicLSTMCell(num_units=hidden_size)
        drop = rnn.DropoutWrapper(cell=lstm)
        return drop

    mlstm_cell = rnn.MultiRNNCell([lstm_cell() for _ in range(layer_num)])

    # 初始化多层LSTM
    init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # 进行LSTM
    outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=input_X, initial_state=init_state, time_major=False)
    return outputs, state


# def vec2onehot(vector, batch_size, Max_Len, Max_Width):
#     ret = tf.Variable(tf.zeros(shape=[batch_size, Max_Len, Max_Width]))
#     temp = tf.one_hot(vector, Max_Width, 1, 0)
#     for i in range(batch_size):


def train(batch_x, batch_Y, batch_size):
    '''
    对训练集进行预测的整体流程
    :param batch_x: 
    :param batch_y: 
    :return: 
    '''

    print('conv')
    # 卷积层
    conv = conv_layer(batch_x)
    print('LSTM')
    # LSTM层
    lstm, state = lstm_layer(conv, batch_size)
    print('conn')
    # 全连接层
    conn = connect_layer(lstm, LONGEST_CH)
    # # 全连接层输出转换为one-hot
    # conn_oh = vec2onehot(conn, LONGEST_CH)

    # 输出值
    out = conn

    print('tf.shape(conn)', tf.shape(conn))
    print('tf.shape(batch_y)', tf.shape(batch_Y))

    onehot_out = vec2onehot_tensor(out)
    onehot_batch_Y = vec2onehot_tensor(batch_Y)

    print('onehot_out',onehot_out)
    print('onehot_batch_Y', onehot_batch_Y)

    # 通过损失函数计算误差
    #loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=onehot_out, labels=onehot_batch_Y))
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels=batch_Y))
    # 定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)

    max_idx_p = out
    max_idx_l = batch_y

    print('max_idx_p', tf.shape(max_idx_p))
    print('max_idx_l', tf.shape(max_idx_l))
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return optimizer, loss, accuracy, out

if __name__ == '__main__':
    with tf.Session() as sess:
        optimizer, loss, acc, out_pre = train(X, Y, BATCH_SIZE)
        sess.run(tf.global_variables_initializer())
        step = 0

        while True:
            _, loss_, out_pre_ = sess.run([optimizer, loss, out_pre],
                                feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75, train_phase: True})

            print("step: %d, loss: %f" % (step, loss_))

            # out_predict = 'max_idx_p'
            # out_correct = 'max_idx_l'
            #
            # with open("%s.txt" % out_predict, "a") as f:
            #     f.write("\n-------------------------------------矩阵形式-----------------------------------------\n")
            #     f.write(str(out_pre_))
            #     f.write("\n-------------------------------------数据形式-----------------------------------------\n")
            #     f.write(str(num_out))
            #
            # with open("%s.txt" % out_correct, "a") as f:
            #     f.write("\n-------------------------------------矩阵形式-----------------------------------------\n")
            #     f.write(str(batch_y))
            #     f.write("\n-------------------------------------数据形式-----------------------------------------\n")
            #     f.write(str(num_predict))

            # 每100 step计算一次准确率
            if step % 100 == 0 and step != 0:
                batch_x_test, batch_y_test = PicCreater.getNextBatchMem(64, IMAGE_WIDTH, IMAGE_HEIGHT)
                #batch_x_test, batch_y_test = PicReader.getNextBatch(Config.PICPATH, BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT)
                acc = sess.run(acc,
                               feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.0, train_phase: False})
                print("step:%d, acc:%f" % (step, acc))
            # 如果准确率大80%,保存模型,完成训练
            # if acc > 0.8:
            #	saver.save(sess, "crack_capcha.model", global_step=step)
            #	break
            step += 1
        
        
