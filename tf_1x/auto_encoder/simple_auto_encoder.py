import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_1x.utilities.file_util import FileUtil

from tensorflow.examples.tutorials.mnist import input_data

"""
实现一个简单自编码器
Input: mnist datasets
    X: [None, 784] (784=28*28*1)
    Y: [None, 10]
网络结构:
1. Encoder
2. Encoder
3. Decoder
4. Decoder

Input:MNIST(784) -> Encoder_1 -> Encoder_2 -> Decoder_1  -> Decoder_2 -> Output: data(784)
(784) -> (256) -> (128) -> (128) -> (256) -> (784)
"""

# 导入MNINST数据集
mnist_data_path = os.path.join(FileUtil.get_datasets_root_path(), "mnist", "input_data")
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

learning_rate = 0.01

# hidden layer settings
n_hidden_1 = 256  # 第一层256个节点
n_hidden_2 = 128  # 第二层128个节点
n_input = 28 * 28 * 1  # MNIST 数据集中图片的维度

# 占位符
x = tf.placeholder("float", [None, n_input])  # 输入
y = x  # 输出

# 学习参数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.zeros([n_input])),
}


# 编码
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))

    # 为了便于编码层的输出，编码层随后一层不使用激活函数
    return layer_2


# 解码
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))

    return layer_2


# 输出的节点
encoder_out = encoder(x=x)
pred = decoder(x=encoder_out)

# cost为y与pred的平方差
cost = tf.reduce_mean(tf.pow(y - pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# 训练参数
training_epochs = 20  # 一共迭代20次
batch_size = 256  # 每次取256个样本
display_step = 5  # 迭代100次输出一次信息

# 定义一个初始化变量的op
init_op = tf.global_variables_initializer()

# 启动会话
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(mnist.train.num_examples / batch_size)
    print(f"[LENGTH] Training set:{mnist.train.num_examples}, Batch size: {batch_size}, Batch count: {total_batch}")

    # 开始训练
    for epoch in range(training_epochs):  # 迭代
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size=batch_size)  # 取数据
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print(f"Epoch:{epoch + 1}, Cost:{round(float(c), 6)}")

    print("Completed!")

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # 计算错误率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", 1 - accuracy.eval({x: mnist.test.images, y: mnist.test.images}))

    # 可视化结果
    show_num = 10
    reconstruction = sess.run(pred, feed_dict={x: mnist.test.images[:show_num]})
    # 将样本对应的自编码重建图像一并输出比较
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(reconstruction[i], (28, 28)))

    plt.show()
