# Make a Two Layer NN to train MNIST
# @Author: CloudCver
# @Date:   2019-03-05
# @E-mail: flying_jun@outlook.com
import numpy as np  
import tensorflow as tf 

from tensorflow.examples.tutorials.mnist import input_data


INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

TRAINING_STEPS = 2000
BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARZATION_RATE= 0.0001


#############################
#训练函数
#
def train(minist):

    #输入数据
    X = tf.placeholder(tf.float32,[None,INPUT_NODE],name = "x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name = "y-output")
    #隐藏层参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev = 0.1))
    b1 = tf.Variable(tf.constant(0.1,shape = [LAYER1_NODE]))
    #输出层参数
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev = 0.1))
    b2 = tf.Variable(tf.constant(0.1,shape = [OUTPUT_NODE]))
    #定义step
    global_step = tf.Variable(0,trainable=False)

    #前向传播
    #计算得分
    layer1 = tf.nn.relu(tf.matmul(X,weight1) + b1)
    y = tf.matmul(layer1,weight2) + b2
    #计算损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #正则化损失
    regular = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    regularization = regular(weight1) + regular(weight2)
    #总损失
    loss = cross_entropy_mean + regularization

    #学习率更新
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #测试
    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    show_result = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



mnist = input_data.read_data_sets('./', one_hot=True)
#mnist = input_data.read_data_sets("/",one_hot = True)

print(mnist.train.num_examples)
print(mnist.train.images[0])

train(mnist)


