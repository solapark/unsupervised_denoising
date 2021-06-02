#import model
import Unet
import time
import imageio
from utils import *
import copy

class Test(object):
    def __init__(self, model_path, save_path, conf,  num_of_adaptation):
        methods=['direct', 'direct', 'bicubic', 'direct']
        self.save_results=True
        self.max_iters=num_of_adaptation
        self.display_iter = 1

        self.back_projection=False
        self.back_projection_iters=4

        self.model_path=model_path
        self.save_path=save_path

        self.build_network(conf)

    def build_network(self, conf):
        tf.reset_default_graph()

        self.lr_decay = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        # Input image
        self.input= tf.placeholder(tf.float32, shape=[None,None,None,3], name='input')
        # Ground truth
        self.label = tf.placeholder(tf.float32, shape=[None,None,None,3],  name='label')
        self.mask = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='mask')
        # parameter variables
        self.PARAM=Unet.Weights(scope='Unet')
        # model class (without feedforward graph)
        self.MODEL = Unet.Unet(name='Unet')
        # Graph build
        self.MODEL.forward(self.input,self.PARAM.weights,1,512,512)
        self.output=self.MODEL.output

        self.loss_t = tf.losses.absolute_difference(self.label*(1-self.mask), self.output*(1-self.mask))

        # Optimizer
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr_decay).minimize(self.loss_t)
        self.init = tf.global_variables_initializer()

        # Variable lists
        self.var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Unet')

        self.loader=tf.train.Saver(var_list=self.var_list)

        self.sess=tf.Session(config=conf)

    def initialize(self):
        self.sess.run(self.init)

        self.loader.restore(self.sess, self.model_path)
        print('=============== Load Meta-trained Model parameters... ==============')

        self.loss = [None] * self.max_iters
        self.mse, self.mse_rec, self.interp_mse, self.interp_rec_mse, self.mse_steps = [], [], [], [], []
        self.psnr=[]
        self.iter = 0

    def __call__(self, img, gt, img_name):
        self.img=img
        self.gt = gt

        self.img_name=img_name

        print('** Start Adaptation for',  os.path.basename(self.img_name), ' **')
        # Initialize network
        self.initialize()


        self.output_shape =self.img.shape[0:2]

        # Train the network
        self.quick_test()

        print('[*] Baseline ')
        self.train()

        post_processed_output = self.final_test()

        if self.save_results:
            if not os.path.exists('%s/%02d' % (self.save_path, self.max_iters)):
                os.makedirs('%s/%02d' % (self.save_path, self.max_iters))

            imageio.imsave('%s/%02d/%s.png' % (self.save_path, self.max_iters, os.path.basename(self.img_name)[:-4]),
                                  post_processed_output)

        print('** Done Adaptation for ', os.path.basename(self.img_name),', PSNR: %.4f' % self.psnr[-1], ' **')
        print('')

        return post_processed_output, self.psnr

    def train(self):
        self.hr_father = self.img
  #      self.lr_son = imresize(self.img, scale=1/self.scale, kernel=self.kernel, ds_method=self.ds_method)
    #    self.lr_son = np.clip(self.lr_son + np.random.randn(*self.lr_son.shape) * self.noise_level, 0., 1.)
        self.lr_son ,self. mask_array = self.generate_mask(copy.deepcopy(self.img))

        t1=time.time()
        for self.iter in range(self.max_iters):


            if self.iter==0:
                self.learning_rate=2e-1
            elif self.iter < 4:
                self.learning_rate=1e-1
            else:
                self.learning_rate=5e-2

            self.train_output = self.forward_backward_pass(self.lr_son, self.hr_father, self.mask_array)

            # Display information
            if self.iter % self.display_iter == 0:
                print('iteration: ', (self.iter+1), ', loss: ', self.loss[self.iter])

            # Test network during adaptation

            # if self.iter % self.display_iter == 0:
            #     output=self.quick_test()

            # if self.iter==0:
            #     imageio.imsave('%s/%02d/01/%s.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4]), output)
            # if self.iter==9:
            #     imageio.imsave('%s/%02d/10/%s_%d.png' % (self.save_path, self.method_num, os.path.basename(self.img_name)[:-4], self.iter), output)

        t2 = time.time()
        print('%.2f seconds' % (t2 - t1))

    def forward_pass(self, input):
        ILR = input

        feed_dict = {self.input : ILR[None,:,:,:]}

        output_=self.sess.run(self.output, feed_dict)
        return np.clip(np.squeeze(output_), 0., 1.)

    def forward_backward_pass(self, input, hr_father, mask):
        ILR = input

        HR = hr_father[None, :, :, :]
        Mask = mask[None,:, :, :]
        # Create feed dict
        feed_dict = {self.input: ILR[None,:,:,:], self.label: HR, self.mask : Mask, self.lr_decay: self.learning_rate}

        # Run network
        _, self.loss[self.iter], train_output = self.sess.run([self.opt, self.loss_t, self.output], feed_dict=feed_dict)
        return np.clip(np.squeeze(train_output), 0., 1.)

 #   def hr2lr(self, hr):
 #       lr = imresize(hr, 1.0 / self.scale, kernel=self.kernel, ds_method=self.ds_method)
 #       return np.clip(lr + np.random.randn(*lr.shape) * self.noise_level, 0., 1.)

    def quick_test(self):
        # 1. True MSE
        self.sr = self.forward_pass(self.img)

        self.mse = self.mse + [np.mean((self.gt - self.sr)**2)]

        '''Shave'''

 #       PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
 #                 rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8))[scale:-scale, scale:-scale])

        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)), rgb2y(np.round(np.clip(self.sr*255., 0., 255.)).astype(np.uint8)))
        self.psnr.append(PSNR)

        # 2. Reconstruction MSE
#        self.reconstruct_output = self.forward_pass(self.hr2lr(self.img), self.img.shape)
#        self.mse_rec.append(np.mean((self.img - self.reconstruct_output)**2))

#        processed_output=np.round(np.clip(self.sr*255, 0., 255.)).astype(np.uint8)

 #       print('iteration: ', self.iter, 'recon mse:', self.mse_rec[-1], ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)
        print('iteration: ', self.iter, ', true mse:', (self.mse[-1] if self.mse else None), ', PSNR: %.4f' % PSNR)
 #       return processed_output

    def final_test(self):

        output = self.forward_pass(self.img)
 #       if self.back_projection == True:
 #           for bp_iter in range(self.back_projection_iters):
 #               output = back_projection(output, self.img, down_kernel=self.kernel,
 #                                                 up_kernel=self.upscale_method, sf=self.scale, ds_method=self.ds_method)

        processed_output=np.round(np.clip(output*255, 0., 255.)).astype(np.uint8)

        '''Shave'''
 #       scale=int(self.scale)
 #       PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8))[scale:-scale, scale:-scale],
 #                 rgb2y(processed_output)[scale:-scale, scale:-scale])

        PSNR=psnr(rgb2y(np.round(np.clip(self.gt*255., 0.,255.)).astype(np.uint8)),
                   rgb2y(processed_output))

        self.psnr.append(PSNR)

        return processed_output

    def generate_mask(self, input,ratio = 0.9, size_window=(5,5)):



#        size_data = self.size_data
        size_data = input.shape
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

        for ich in range(size_data[2]):
#        for ich in range(1):
            idy_msk = np.random.randint(0, size_data[0], num_sample)
            idx_msk = np.random.randint(0, size_data[1], num_sample)

            idy_neigh = np.random.randint(-size_window[0] // 2 + size_window[0] % 2,
                                          size_window[0] // 2 + size_window[0] % 2, num_sample)
            idx_neigh = np.random.randint(-size_window[1] // 2 + size_window[1] % 2,
                                          size_window[1] // 2 + size_window[1] % 2, num_sample)

            idy_msk_neigh = idy_msk + idy_neigh
            idx_msk_neigh = idx_msk + idx_neigh

            idy_msk_neigh = idy_msk_neigh + (idy_msk_neigh < 0) * size_data[0] - (idy_msk_neigh >= size_data[0]) * \
                            size_data[0]
            idx_msk_neigh = idx_msk_neigh + (idx_msk_neigh < 0) * size_data[1] - (idx_msk_neigh >= size_data[1]) * \
                            size_data[1]

            id_msk = (idy_msk, idx_msk, ich)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

#            id_msk = (idy_msk, idx_msk)
#            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0
			 ######################################
#        mask=np.stack((mask,)*3,axis=-1)
#############################################
        return output, mask

