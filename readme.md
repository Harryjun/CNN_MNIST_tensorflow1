

##  概述

学习完CS231N一系列课程后，我们都书写了自己的深层卷积网络，代码量不多。这里我们使用google的tensorflow框架去写一个卷积神经网络，并去把卷积网路中的优化一步一步复现出来。

搭建一个卷积神经网络，需要包括以下几个部分：

 1. **数据输入**
 2. **前向传播**
 3. **损失计算+正则化优化**
 4. **网络优化**
 5. **测试**
 
##  一、数据输入
数据输入，定义变量我们定义了两个输入数据的占位，方便之后运行时传入即可。然后定义了隐含层的参数输出层的参数。
```python
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
```

##  二、前向传播
前向传播比价简单
1、矩阵相乘matmul
```python
	a = tf.matmul(X,weight1)
```
2、激活函数
激活函数有很多种类
RELU激活函数tf.nn.relu()
```python
    layer1 = tf.nn.relu(tf.matmul(X,weight1) + b1)
```
3、一个两层的神经网络
代码如下：
```python
	#前向传播
    #计算得分
    layer1 = tf.nn.relu(tf.matmul(X,weight1) + b1)
    y = tf.matmul(layer1,weight2) + b2
```

## 三、损失计算+正则化优化
经过前向传播我们得到了传播后的分数，之后我们需要计算它的损失，一部分是得分的损失，一部分是权重的正则损失。

### softmax回归+交叉熵损失 
在计算损失之前，我们先对结果进行分类回归，一般对于多分类问题，我们采用softmax分类器进行回归，softmax可以将分数转变为概率的形式，就可以反应每个数据归属于每个类别的概率。然后我们采用交叉熵（cross_entropy）来计算损失。
tensorflow中提供了tf.nn.sparse_softmax_cross_entropy_with_logits函数来计算损失，其输入有两个参数logits是回归结果，label是标签。
logits填入待取log的概率化数据。格式为[N,C]，其中N代表样本数量，C代表最后的划分类别数。
label填入标签数据，格式为[N,1]，其中N代表样本数量，每一行只有一个结果，取值0-C。
这里我们用的数据集样本标签和logits格式是一样的，所以我们引用tf.argmax选取每一行的最大值所代表的编号，得到[N,1]类型结果。
计算完后我们将所有损失加起来取平均。如下所示：
```python
	#计算损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
```
###  正则化
有时候我们的数据量少，参数多的时候可能会出现过拟合现象，我们一般采取正则化的方式来优化，过拟合的原因在于权重参数可能在调解过程中变得过于复杂去适应我们的训练数据，就好比我们有五个不等式方程，我们去改善它的5*5个系数，使他对于我们的数据都能正确，如果一次取样10个数据，我们可能获得等式的25个唯一参数，很好的去适应这10个数据，但换一组数据它就不是那么好，在控制中就是鲁棒性不是太好。
所以我们引入一个刻画权重参数复杂程度的指标J(θ)，然后用一个系数乘以它，来调节它对整个损失的影响比例。这就是正则化
一般我们采用L1、L2损失函数，在tensorflow中为：
```python
    regular = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)(weight)
```
把L2改为L1就代表L1正则化。可以看出后面有两个括号，第一个括号参数为正则化系数，第二个为权重。也可以改写为下面的代码，先声明一个具有某个正则化系数的函数regular，然后在计算W1与W2时可以直接用regular(W1)。
```python
    #正则化损失
    regular = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    regularization = regular(weight1) + regular(weight2)
```
PS：两个正则化的对比：L1正则化会让参数变得稀疏（指会有更多参数变为0），而L2正则化则不会（因为参数的平方后会让小的参数变得更小，大的参数变得更大，同样起到了特征选取的功能，而不会让参数变为0）。其次是L1正则化计算不可导，而L2的正则化损失函数可导。

下面给出整体的损失计算代码
```python
	#计算损失
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y, labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #正则化损失
    regular = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    regularization = regular(weight1) + regular(weight2)
    #总损失
    loss = cross_entropy_mean + regularization
```


## 四、网络优化Optimizer
计算完损失之后需要进行网络优化，网络优化也就是通过不停改变参数把损失值降低，一般我们多采用随机梯度下降法，

###  Optimizer优化
1、梯度下降法
一般我们常用的优化算法为梯度下降法，也就是每次计算出梯度后对每一个参数进行线形调节X = X + learning_rate * Gradient
```python
tf.train.GradientDescentOptimizer(learning_rate)#梯度下降法类
```
这种优化算法存在一些问题，可能会出现有好几个波谷，但是梯度下降一直在一个波谷里面来回动荡。这会导致梯度下降法有时不能找到最优解，只能找到一个极值点。再者梯度下降法对所有数据进行运算，耗时太长。
每一个优化方法类下都有一个minimize函数，对某个值进行最小化优化。
```python
train_ = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
```

2、随机梯度下降法（SGD）
考虑梯度下降法优化速度慢的问题，后来有人提出随机梯度下降法，我们每次训练可以不训练所有的数据，可以只取一个小的batch数据进行训练，这样每次训练的速度就会加快，然后我们每次取样的batch不一样，这样就保证稳定性了。
代码可以和上面一样，考虑每次训练取样batch即可。

更多的优化算法在我另一篇博客中介绍。

在这之后我们在考虑一个学习率的问题，学习率在一开始可能需要大一点比价好，加快收敛速度，但是在后期，就需要慢下来，因为幅度大的话可能在两边来回晃，很难收敛。
###  学习率优化问题
这里给出一个**指数衰减法**的方法。让学习率以指数方式慢慢衰减。这里我抓取一张常用的图，tensorflow提供两个衰减情况，一个是连续衰减，一个是梯度衰减，如下图所示。
![指数衰减法](https://img-blog.csdnimg.cn/20190316213621316.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NMT1VEX0o=,size_16,color_FFFFFF,t_70)
tensorflow给出以下函数expoential_decay，参数有四个，基础学习率，全局步数，衰减步数，衰减率。其公式为：
learning_rate = LEARNING_RATE_BASE * LEARNING_RATE_DECAY^(global_step/decay_step)
含义很明白，当全局步数达到衰减步数时学习率变为基础学习率的基础上乘以衰减率，如果是连续衰减时，在这过程中则一直变化。
```python
#学习率更新
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,\
   decay_step,LEARNING_RATE_DECAY)
```



## 五、测试结果
在进行一段时间训练后我们希望输出以测试集的正确性，利用tf.equal得到两组数据是否i相等，然后求结果的平均值，如下所示。
tf.cast可以将其转换类型，因为equal返回的时bool型。故我们将其转换为float32再求平均数。

```python
	correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    show_result = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
```

## 六、运行
首先开启一个会话，然后初始化所有变量，然后准备一个字典型数据，mnist数据集给了一个取样函数next_batch直接取样即可，然后运行_train把数据送进去。当运行100次时打印数据。
```python
    with tf.Session() as sess:
        #初始化数据
        tf.global_variables_initializer().run()
        #验证集数据
        validate_feed = {X:mnist.validation.images,y_:mnist.validation.labels}

        for it in range(TRAINING_STEPS):
            XS,YS = mnist.train.next_batch(BATCH_SIZE)#取样数据
            sess.run(train_,feed_dict={X:XS,y_:YS})
            if it % 100 == 0:
                #打印数据
                result = sess.run(show_result,feed_dict=validate_feed)
                print("After %d training step(s),validation accuracy is %g"%(it,result))

```
主函数我们编写如下代码：
```python

mnist = input_data.read_data_sets('./', one_hot=True)
#mnist = input_data.read_data_sets("/",one_hot = True)

print(mnist.train.num_examples)
print(mnist.train.images[0])

train(mnist)

```
注意：这里我们是在本地存了一些数据，如有需要可以从我的github获取
[用Tensorflow搭建两层神经网络训练MNIST数据集](https://github.com/Harryjun/CNN_MNIST_tensorflow1)



最后结果如下所示
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190316215633299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0NMT1VEX0o=,size_16,color_FFFFFF,t_70)


