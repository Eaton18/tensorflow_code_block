import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tf_1x.utilities.file_util import FileUtil

from tensorflow.examples.tutorials.mnist import input_data

# 导入MNINST数据集
mnist_data_path = os.path.join(FileUtil.get_datasets_root_path(), "mnist", "input_data")
mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

train_X = mnist.train.images
train_Y = mnist.train.labels
test_X = mnist.test.images
test_Y = mnist.test.labels

# tf.reset_default_graph()

n_input = 784
n_hidden_1 = 256

# 占位符
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_input])
dropout_keep_prob = tf.placeholder("float")  # for dropout layer

# 学习参数
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
}

biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_1])),
    'out': tf.Variable(tf.zeros([n_input]))
}


# 网络模型
def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):
    """

    :param _X:
    :param _weights:
    :param _biases:
    :param _keep_prob:
    :return:
    """
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)

    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])


reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)

# COST
cost = tf.reduce_mean(tf.pow(reconstruction - y, 2))
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)

# 训练参数
epochs = 10
batch_size = 256
disp_step = 1

# 定义一个初始化变量的op
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    # 开始训练
    for epoch in range(epochs):
        num_batch = int(mnist.train.num_examples / batch_size)
        total_cost = 0
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 加入噪声，将输入图像的每一个像素都加上0.3倍的高斯噪声
            batch_xs_noisy = batch_xs + 0.3 * np.random.randn(batch_size, 784)
            feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)

        # 显示训练日志
        if epoch % disp_step == 0:
            print(f"Epoch {epoch}/{epochs}  Average cost: {round(total_cost / num_batch, 6)}")

    print("completed!")

    correct_prediction = tf.equal(tf.argmax(reconstruction, 1), tf.argmax(y, 1))
    # 计算错误率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", 1 - accuracy.eval({x: mnist.test.images, y: mnist.test.images, dropout_keep_prob: 1.}))

    # 数据可视化

    # 测试增加相同噪声的方法的结果
    if False:
        show_num = 10
        test_noisy = mnist.test.images[:show_num] + 0.3 * np.random.randn(show_num, 784)
        encode_decode = sess.run(reconstruction, feed_dict={x: test_noisy, dropout_keep_prob: 1.})
        f, a = plt.subplots(4, 10, figsize=(10, 4))
        for i in range(show_num):
            # 加入噪声后的数据
            a[0][i].imshow(np.reshape(test_noisy[i], (28, 28)))
            # 测试集
            a[1][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            # 降噪后的数据
            a[2][i].imshow(np.reshape(encode_decode[i], (28, 28)))
            # 灰色映射，变成黑白图像
            a[3][i].matshow(np.reshape(encode_decode[i], (28, 28)), cmap=plt.get_cmap('gray'))

        plt.show()

    # 测试鲁棒性，换一种噪声测试一个
    if True:
        randidx = np.random.randint(test_X.shape[0], size=1)
        orgvec = test_X[randidx, :]
        testvec = test_X[randidx, :]
        label = np.argmax(test_Y[randidx, :], 1)  # 将onehot编码转成一般编码

        print("label is %d" % (label))

        # Noise type
        print("Salt and Pepper Noise")
        noisyvec = testvec
        rate = 0.15
        noiseidx = np.random.randint(test_X.shape[1], size=int(test_X.shape[1] * rate))
        noisyvec[0, noiseidx] = 1 - noisyvec[0, noiseidx]

        outvec = sess.run(reconstruction, feed_dict={x: noisyvec, dropout_keep_prob: 1})
        outimg = np.reshape(outvec, (28, 28))

        # Plot
        plt.matshow(np.reshape(orgvec, (28, 28)), cmap=plt.get_cmap('gray'))
        plt.title("Original Image")
        plt.colorbar()

        plt.matshow(np.reshape(noisyvec, (28, 28)), cmap=plt.get_cmap('gray'))
        plt.title("Input Image")
        plt.colorbar()

        plt.matshow(outimg, cmap=plt.get_cmap('gray'))
        plt.title("Reconstructed Image")
        plt.colorbar()
        plt.show()
