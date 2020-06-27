import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_1x.utilities.file_util import FileUtil

"""
实现一个卷积神经网络
input: mnist datasets
    X: [None, 784]
    Y: [None, 10]
网络结构:
1. conv
    卷积核: 32个, 5*5, strides=1, padding="SAME", bias=32
    input: [None, 28, 28, 1]
    output: [None, 28, 28, 32]
2. relu
    input: [None, 28, 28, 32]
    output: [None, 28, 28, 32]
3. pooling
    2*2, strides=2, padding="SAME"
    input: [None, 28, 28, 32]
    output: [None, 14, 14, 32]
4. conv
    卷积核: 64个, 5*5, strides=1, padding="SAME", bias=64
    input: [None, 14, 14, 32]
    output: [None, 14, 14, 64]
5. relu
    input: [None, 14, 14, 64]
    output: [None, 14, 14, 64]
6. pooling
    2*2, strides=2
    input: [None, 14, 14, 64]
    output: [None, 7, 7, 64]
7. full connected
    weight=[7*7*64, 10], bias=10
    input:  [None, 7, 7, 64] ->  [None, 7*7*64]
    output: [None, 10]
"""


# 定义一个初始化权重的函数
def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w


# 定义一个初始化偏置的函数
def bias_variables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b


def model():
    """
    自定义的卷积模型
    :return:
    """
    # 1、准备数据
    # 准备占位符(x [None, 784]  y_true [None, 10])
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2、卷积层一
    # 卷积: 5*5*1，32个，strides=1 激活: tf.nn.relu 池化
    with tf.variable_scope("conv1"):
        # 随机初始化权重, 偏置[32]
        # [5, 5, 1, 32], [长，宽，通道数（黑_1 or 彩色_3），filter个数]
        w_conv1 = weight_variables([5, 5, 1, 32])

        b_conv1 = bias_variables([32])

        # 对x进行形状的改变[None, 784] -> [None, 28, 28, 1]
        # reshape 时不知道样本数量时不能填None，填-1
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # Conv: [None, 28, 28, 1] -> [None, 28, 28, 32]
        x_conv1 = tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME")

        # ReLU: [None, 28, 28, 32] -> [None, 28, 28, 32]
        x_relu1 = tf.nn.relu(x_conv1 + b_conv1)

        # Pooling: 2*2, strides2 [None, 28, 28, 32]---->[None, 14, 14, 32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、卷积层二 卷积: 5*5*32，64个filter，strides=1 激活: tf.nn.relu 池化：
    with tf.variable_scope("conv2"):
        # 随机初始化权重，偏置[64]
        # [5, 5, 32, 64]
        w_conv2 = weight_variables([5, 5, 32, 64])

        b_conv2 = bias_variables([64])

        # 卷积，激活，池化计算
        # [None, 14, 14, 32]-----> [None, 14, 14, 64]
        x_conv2 = tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME")
        x_relu2 = tf.nn.relu(x_conv2 + b_conv2)

        # 池化 2*2, strides 2, [None, 14, 14, 64]---->[None, 7, 7, 64]
        x_pool2 = tf.nn.max_pool(x_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 4、全连接层 [None, 7, 7, 64]--->[None, 7*7*64]*[7*7*64, 10]+ [10] =[None, 10]
    with tf.variable_scope("conv2"):
        # 随机初始化权重和偏置
        w_fc = weight_variables([7 * 7 * 64, 10])

        b_fc = bias_variables([10])

        # 修改形状 [None, 7, 7, 64] --->None, 7*7*64]
        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])

        # 进行矩阵运算得出每个样本的10个结果
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    # 获取mnist数据集
    mnist_data_path = os.path.join(FileUtil.get_datasets_root_path(), "mnist", "input_data")
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

    # 定义模型，得出输出
    x, y_true, y_predict = model()

    # 进行交叉熵损失计算
    # 3、求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):
        # 求平均交叉熵损失# 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降求出损失
    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

    # 5、计算准确率
    with tf.variable_scope("acc"):
        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        # equal_list  None个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 开启回话运行
    with tf.Session() as sess:
        sess.run(init_op)

        train_len = 2000
        # 循环去训练
        for i in range(train_len):
            # 取出真实存在的特征值和目标值
            mnist_x, mnist_y = mnist.train.next_batch(50)

            # 运行train_op训练
            sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

            accu_val = round(sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y}), 4)
            print(f"STEP {i}, Accuracy:{accu_val}")

    return None


if __name__ == "__main__":
    conv_fc()
