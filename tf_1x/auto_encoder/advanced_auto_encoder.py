import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_1x.utilities.file_util import FileUtil

# 导入 MINST 数据集
from tensorflow.examples.tutorials.mnist import input_data

# 导入MNINST数据集
mnist_data_path = os.path.join(FileUtil.get_datasets_root_path(), "mnist", "input_data")
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

"""
实现一个带有线性解码器的自编码器
由多个带有S型激活函数的隐含层及一个线性输出层构成的自编码器，称为线性解码器。
Input: MNIST datasets
    X: [None, 784] (784=28*28*1)
    Y: [None, 10]
    
网络结构:
1. Encoder 256
2. Encoder 64
3. Encoder 16
4. Encoder 2

5. Decoder 2
6. Decoder 16
7. Decoder 64
8. Decoder 256

Input:MNIST(784) -> Encoder_1 -> Encoder_2 -> Encoder_3 -> Encoder_4 -> 
Decoder_1 -> Decoder_2 -> Decoder_3 -> Decoder_4 -> Output: data(784)
(784) -> (256) -> (64) -> (16) -> (2) -> 
(2) -> (16) -> (64) -> (256) -> (784)
"""

# 参数设置
learning_rate = 0.01
# hidden layer settings
n_hidden_1 = 256
n_hidden_2 = 64
n_hidden_3 = 16
n_hidden_4 = 2
n_input = 784  # MNIST data 输入 (img shape: 28*28)

# 占位符
x = tf.placeholder("float", [None, n_input])
y = x

# 学习参数
weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], )),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], )),
    'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], )),
    'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], )),

    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3], )),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2], )),
    'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1], )),
    'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input], )),
}

biases = {
    'encoder_b1': tf.Variable(tf.zeros([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.zeros([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.zeros([n_hidden_4])),

    'decoder_b1': tf.Variable(tf.zeros([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.zeros([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.zeros([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.zeros([n_input])),
}


def encoder(x):
    """
    编码阶段
    编码的最后一层，使用了线性解码器，没有进行Sigmoid变换
    这是因为生成的二维数据其数据特征已经变得极为主要，所以希望让它传到解码器中， 少一些变换可以最大化地保存原有的主要特征。
    :param x:
    :return:
    """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']), biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']), biases['encoder_b4'])
    return layer_4


def decoder(x):
    """
    解码阶段
    :param x:
    :return:
    """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']), biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']), biases['decoder_b4']))
    return layer_4


# 构建模型
encoder_op = encoder(x)
y_pred = decoder(encoder_op)  # 784 Features

cost = tf.reduce_mean(tf.pow(y - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 训练
training_epochs = 20  # 20 Epoch 训练
batch_size = 256
display_step = 1

# 定义一个初始化变量的op
init_op = tf.global_variables_initializer()

# 启动会话
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(mnist.train.num_examples / batch_size)
    # 启动循环开始训练
    for epoch in range(training_epochs):
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs})

        # 显示训练中的详细信息
        if epoch % display_step == 0:
            print(f"Epoch:{epoch + 1}, Cost:{round(float(c), 6)}")

    print("Completed!")

    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    # 计算错误率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", 1 - accuracy.eval({x: mnist.test.images, y: mnist.test.images}))

    # 可视化结果
    show_num = 10
    encode_decode = sess.run(y_pred, feed_dict={x: mnist.test.images[:show_num]})
    # 将样本对应的自编码重建图像一并输出比较
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(show_num):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
    plt.show()

    aa = [np.argmax(l) for l in mnist.test.labels]  # 将onehot编码转成一般编码
    encoder_result = sess.run(encoder_op, feed_dict={x: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=aa)  # mnist.test.labels)
    plt.colorbar()
    plt.show()
