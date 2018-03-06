import tensorflow as tf

def createVariable():
    w_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32]), name='w_conv1')
    return w_conv1

if __name__ == '__main__':
    with tf.Session() as sess:
        test = createVariable()
        sess.run(tf.global_variables_initializer())
        ret = sess.run(test)
        print(str(ret))