from ops import *

class Weights(object):
    def __init__(self, scope=None):
        self.weights={}
        self.scope=scope
        self.kernel_initializer=tf.variance_scaling_initializer()

        self.build_CNN_params()
        print('Initialize weights {}'.format(self.scope))


    def build_CNN_params(self):
        kernel_initializer=self.kernel_initializer
        with tf.variable_scope(self.scope):
            self.weights['conv1/w'] = tf.get_variable('conv1/kernel', [3, 3, 3, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv1/b'] = tf.get_variable('conv1/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv2/w'] = tf.get_variable('conv2/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv2/b'] = tf.get_variable('conv2/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv3/w'] = tf.get_variable('conv3/kernel', [3, 3, 64, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv3/b'] = tf.get_variable('conv3/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv4/w'] = tf.get_variable('conv4/kernel', [3, 3, 128,128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv4/b'] = tf.get_variable('conv4/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv5/w'] = tf.get_variable('conv5/kernel', [3, 3, 128, 256], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv5/b'] = tf.get_variable('conv5/bias',[256], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv6/w'] = tf.get_variable('conv6/kernel', [3, 3, 256, 256], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv6/b'] = tf.get_variable('conv6/bias',[256], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv7/w'] = tf.get_variable('conv7/kernel', [3, 3, 256, 512], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv7/b'] = tf.get_variable('conv7/bias',[512], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv8/w'] = tf.get_variable('conv8/kernel', [3, 3, 512,512], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv8/b'] = tf.get_variable('conv8/bias',[512], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv9/w'] = tf.get_variable('conv9/kernel', [3, 3, 512, 1024], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv9/b'] = tf.get_variable('conv9/bias',[1024], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv10/w'] = tf.get_variable('conv10/kernel', [3, 3, 1024,1024], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv10/b'] = tf.get_variable('conv10/bias',[1024], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv11/w'] = tf.get_variable('conv11/kernel', [2,2,512,1024], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv11/b'] = tf.get_variable('conv11/bias',[512], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv12/w'] = tf.get_variable('conv12/kernel', [3,3,1024,512], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv12/b'] = tf.get_variable('conv12/bias',[512], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv13/w'] = tf.get_variable('conv13/kernel', [3, 3, 512, 512], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv13/b'] = tf.get_variable('conv13/bias',[512], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv14/w'] = tf.get_variable('conv14/kernel', [2, 2, 256, 512], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv14/b'] = tf.get_variable('conv14/bias',[256], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv15/w'] = tf.get_variable('conv15/kernel', [3, 3, 512, 256], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv15/b'] = tf.get_variable('conv15/bias',[256], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv16/w'] = tf.get_variable('conv16/kernel', [3, 3, 256, 256], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv16/b'] = tf.get_variable('conv16/bias',[256], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv17/w'] = tf.get_variable('conv17/kernel', [2, 2, 128, 256], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv17/b'] = tf.get_variable('conv17/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv18/w'] = tf.get_variable('conv18/kernel', [3, 3, 256, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv18/b'] = tf.get_variable('conv18/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())


            self.weights['conv19/w'] = tf.get_variable('conv19/kernel', [3, 3, 128, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv19/b'] = tf.get_variable('conv19/bias',[128], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv20/w'] = tf.get_variable('conv20/kernel', [2, 2, 64, 128], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv20/b'] = tf.get_variable('conv20/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv21/w'] = tf.get_variable('conv21/kernel', [3, 3, 128, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv21/b'] = tf.get_variable('conv21/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv22/w'] = tf.get_variable('conv22/kernel', [3, 3, 64, 64], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv22/b'] = tf.get_variable('conv22/bias',[64], dtype=tf.float32, initializer=tf.zeros_initializer())

            self.weights['conv23/w'] = tf.get_variable('conv23/kernel', [3, 3, 64, 3], initializer=kernel_initializer, dtype=tf.float32)
            self.weights['conv23/b'] = tf.get_variable('conv23/bias',[3], dtype=tf.float32, initializer=tf.zeros_initializer())

class Unet(object):
    def __init__(self, name):
        self.name = name

        print('Build Model {}'.format(self.name))

    def forward(self, x, param, batch_size, img_size_h, img_size_w, train=True):
        self.input=x
        self.param=param
        self.batch_size=batch_size
        self.img_size_w, self. img_size_h = img_size_h , img_size_w

        with tf.variable_scope(self.name):
            self.conv1 = conv2d(self.input, param['conv1/w'], param['conv1/b'], scope='conv1', activation='ReLU')
            self.conv2 = conv2d(self.conv1, param['conv2/w'], param['conv2/b'], scope='conv2', activation='ReLU')
            self.pool2=pool2d(self.conv2)
            self.conv3 = conv2d(self.pool2, param['conv3/w'], param['conv3/b'], scope='conv3', activation='ReLU')
            self.conv4 = conv2d(self.conv3, param['conv4/w'], param['conv4/b'], scope='conv4', activation='ReLU')
            self.pool4= pool2d(self.conv4)
            self.conv5 = conv2d(self.pool4, param['conv5/w'], param['conv5/b'], scope='conv5', activation='ReLU')
            self.conv6 = conv2d(self.conv5, param['conv6/w'], param['conv6/b'], scope='conv6', activation='ReLU')
            self.pool6 = pool2d(self.conv6)
            self.conv7= conv2d(self.pool6, param['conv7/w'], param['conv7/b'], scope='conv7', activation='ReLU')
            self.conv8 = conv2d(self.conv7, param['conv8/w'], param['conv8/b'], scope='conv8', activation='ReLU')
            self.pool8 = pool2d(self.conv8)
            self.conv9 = conv2d(self.pool8, param['conv9/w'], param['conv9/b'], scope='conv9', activation='ReLU')
            self.conv10 = conv2d(self.conv9, param['conv10/w'], param['conv10/b'], scope='conv10', activation='ReLU')
            self.conv11 = un_conv2d(self.conv10, param['conv11/w'], param['conv11/b'], output_size_h=self.img_size_h//8,output_size_w=self.img_size_w//8, batch_size=self.batch_size,num_filters=512,stride=2 ,scope='conv11', activation='ReLU',train=train)
            self.merge11 = tf.concat(values=[self.conv8, self.conv11],axis=-1)
            self.conv12 = conv2d(self.merge11, param['conv12/w'], param['conv12/b'], scope='conv12', activation='ReLU')
            self.conv13 = conv2d(self.conv12, param['conv13/w'], param['conv13/b'], scope='conv13', activation='ReLU')
            self.conv14 = un_conv2d(self.conv13, param['conv14/w'], param['conv14/b'], output_size_h=self.img_size_h // 4, output_size_w=self.img_size_w//4,
                                    batch_size=self.batch_size, num_filters=256, stride=2, scope='conv14',
                                    activation='ReLU',train=train)
            self.merge14 = tf.concat(values=[self.conv6, self.conv14], axis=-1)
            self.conv15 = conv2d(self.merge14, param['conv15/w'], param['conv15/b'], scope='conv15', activation='ReLU')
            self.conv16 = conv2d(self.conv15, param['conv16/w'], param['conv16/b'], scope='conv16', activation='ReLU')
            self.conv17 = un_conv2d(self.conv16, param['conv17/w'], param['conv17/b'], output_size_h=self.img_size_h // 2,output_size_w=self.img_size_w // 2,
                                    batch_size=self.batch_size, num_filters=128, stride=2, scope='conv17',
                                    activation='ReLU',train=train)
            self.merge17 = tf.concat(values=[self.conv4, self.conv17], axis=-1)
            self.conv18 = conv2d(self.merge17, param['conv18/w'], param['conv18/b'], scope='conv18', activation='ReLU')
            self.conv19 = conv2d(self.conv18, param['conv19/w'], param['conv19/b'], scope='conv19', activation='ReLU')
            self.conv20 = un_conv2d(self.conv19, param['conv20/w'], param['conv20/b'], output_size_h=self.img_size_h,output_size_w=self.img_size_w,
                                    batch_size=self.batch_size, num_filters=64, stride=2, scope='conv20',
                                    activation='ReLU',train=train)
            self.merge20 = tf.concat(values=[self.conv2, self.conv20], axis=-1)
            self.conv21 = conv2d(self.merge20, param['conv21/w'], param['conv21/b'], scope='conv21', activation='ReLU')
            self.conv22 = conv2d(self.conv21, param['conv22/w'], param['conv22/b'], scope='conv22', activation='ReLU')
            self.output = conv2d(self.conv22, param['conv23/w'], param['conv23/b'], scope='conv23', activation='ReLU')