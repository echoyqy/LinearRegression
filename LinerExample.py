import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

np.random.seed(5)
x_data = np.linspace(-1, 1, 100)
# 输出数据
y_data = 2*x_data+1.0+np.random.rand(*x_data.shape)*0.4
# # 目标值
plt.scatter(x_data, y_data)
# plt.plot(x_data, 2*x_data+1, color='red', linewidth=3)
# plt.show()

# 训练数据的模型
x = tf.placeholder('float', name='x')
y = tf.placeholder('float', name='y')
# 定义模型函数


def model(x, w, b):
    return tf.multiply(x, w) + b


# 模型结构
w = tf.Variable(1.0, name='w0')
b = tf.Variable(1.0, name='b0')
pred = model(x, w, b)
# 迭代次数
train_epoches = 10
# 学习速率
learning_rate = 0.01
# 损失函数
loss_function = tf.reduce_mean(tf.square(y-pred))
# 梯度下降优化器
optizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_function)

# sess对话
sess = tf.Session()

# 运行sess

init = tf.global_variables_initializer()
sess.run(init)

for epoch in range(train_epoches):
    for xs,ys in zip(x_data, y_data):
        _,loss=sess.run([optizer, loss_function], feed_dict={x: xs, y: ys})
    bo_temp = b.eval(session=sess)
    # b.eval返回表示式的结果
    wo_temp = w.eval(session=sess)
    # print(loss)
    # plt.plot (x_data, wo_temp*x_data+bo_temp)
    # plt.show()
    # plt.scatter(x_data, y_data)
    # plt.plot(x_data, sess.run(w)*x_data+sess.run(b), color='red', linewidth=3)
    # plt.show()

# 测试集
x_test = 3.21
predict = sess.run(pred, feed_dict={x: x_test})
print(predict)
value_target = x_test*2+1
print(value_target)







