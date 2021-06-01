import tensorflow as tf

def conv2d(x, kernel, bias, strides=1, scope=None, activation=None):
    with tf.variable_scope(scope):
        out = tf.nn.conv2d(x,kernel,[1,strides,strides,1],padding='SAME', name='conv2d')
        out = tf.nn.bias_add(out,bias, name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)

def dense(x, weights, bias, scope=None, activation=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        out=tf.matmul(x, weights,name='dense')
        out=tf.nn.bias_add(out,bias,name='BiasAdd')

        if activation is None:
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out, 0.2)

def un_conv2d(x, kernel, bias, output_size_h,output_size_w, batch_size, num_filters, stride=2, scope = None, activation= None, train=True):
    if train :
        batch_size_0 = batch_size
    else :
        batch_size_0 = 1
    with tf.variable_scope(scope):
        out = tf.nn.conv2d_transpose(x,kernel, output_shape = [batch_size_0, output_size_h, output_size_w, num_filters], strides = [1,stride,stride,1],padding='SAME', name='unconv2d')
        out = tf.nn.bias_add(out, bias, name='BiasAdd')
        if activation is None :
            return out
        elif activation is 'ReLU':
            return tf.nn.relu(out)
        elif activation is 'leakyReLU':
            return tf.nn.leaky_relu(out,0.2)

def pool2d(x,padding='SAME'):
    return tf.nn.max_pool(value=x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = padding)
