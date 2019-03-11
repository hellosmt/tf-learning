import tensorflow as tf
from numpy.random import RandomState   # 这里只写numpy会报错

batch_size = 8
## 权重 偏置
weights1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
weights2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

## 数据placeholder 这里None是为了方面后面传入指定batch_size大小的数据集
x = tf.placeholder(tf.float32, shape = (None, 2), name='x-input')
y = tf.placeholder(tf.float32, shape = (None, 1), name='y-input')

## 前向传播过程
a = tf.matmul(x, weights1)
y_ = tf.matmul(a, weights2)

## 定义损失函数和反向传播
y = tf.sigmoid(y_)
cross_entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(y_,1e-10,1.0))+(1-y)*tf.log(tf.clip_by_value(1-y_,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

## 生成模拟数据集
rdm = RandomState(1)
dataset_size = 256
X = rdm.rand(dataset_size,2)
Y = [[(int(x1+x2)>1)] for (x1,x2) in X]

## 创建会话执行任务
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op) 

    ## 打印初始的权重值
    print(sess.run(weights1))
    print(sess.run(weights2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size)%dataset_size
        end = min(start+batch_size, dataset_size)

        sess.run(train_step, feed_dict={x:X[start:end], y:Y[start:end]})
        if i%1000==0:
            total_loss = sess.run(cross_entropy, feed_dict={x:X,y:Y})
            print("Steps %d:cross on all data is %g" % (i,total_loss))
    
    ## 训练之后的权重值
    print(sess.run(weights1))
    print(sess.run(weights2))



