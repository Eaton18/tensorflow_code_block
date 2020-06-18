import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from tf_1x.utilities.file_util import FileUtil

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("is_train", 0, "Flag: train or pred")  # 1:training, other: prediction


def full_connected_nn():
    """
    实现一个单层的全连接神经网络
    :return: 
    """

    # 获取真实的数据
    mnist_data_path = os.path.join(FileUtil.get_datasets_root_path(), "mnist", "input_data")
    mnist = input_data.read_data_sets(mnist_data_path, one_hot=True)

    # 1、建立数据的占位符 x [None, 784]    y_true [None, 10]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])

        y_true = tf.placeholder(tf.int32, [None, 10])

    # 2、建立一个全连接层的神经网络 w [784, 10]   b [10]
    with tf.variable_scope("fc_model"):
        # 随机初始化权重和偏置
        weight = tf.Variable(tf.random_normal([784, 10], mean=0.0, stddev=1.0), name="w")

        bias = tf.Variable(tf.constant(0.0, shape=[10]))

        # 预测None个样本的输出结果matrix [None, 784]* [784, 10] + [10] = [None, 10]
        y_predict = tf.matmul(x, weight) + bias

    # 3、求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):

        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降求出损失
    with tf.variable_scope("optimizer"):

        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 5、计算准确率
    with tf.variable_scope("acc"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))

        # equal_list  None个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # 收集变量 单个数字值收集
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("acc", accuracy)

    # 高纬度变量收集
    tf.summary.histogram("weightes", weight)
    tf.summary.histogram("biases", bias)

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个合并变量de op
    merged = tf.summary.merge_all()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话去训练
    with tf.Session() as sess:
        # 初始化变量
        sess.run(init_op)

        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter("./summary/", graph=sess.graph)

        if FLAGS.is_train == 1:

            # 迭代步数去训练，更新参数预测
            for i in range(2000):
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)

                # 运行train_op训练
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                # 写入每步训练的值
                summary = sess.run(merged, feed_dict={x: mnist_x, y_true: mnist_y})

                filewriter.add_summary(summary, i)

                print("Step:%d, Accuracy:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

            # 保存模型
            saver.save(sess, "./checkpoint/fc_model")
        else:
            # 加载模型
            saver.restore(sess, "./checkpoint/fc_model")

            pred_len = 200
            accu_cnt = 0

            # 如果是0，做出预测
            for i in range(pred_len):
                # 每次测试一张图片 [0,0,0,0,0,1,0,0,0,0]
                x_test, y_test = mnist.test.next_batch(1)

                actual_val = tf.argmax(y_test, 1).eval()
                pred_val = tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                print(f"Figure: {i}, Actual value:{actual_val[0]}, Predicted value:{pred_val[0]}")

                if actual_val[0] == pred_val[0]:
                    accu_cnt += 1

            accuracy = round(accu_cnt / pred_len, 4)
            print(f"Accuracy: {accuracy}")

    return None


if __name__ == "__main__":
    full_connected_nn()
