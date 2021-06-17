from utils import *
import copy
#from imresize import imresize
#from gkernel import generate_kernel
import imageio
import glob
import numpy as np
import random
import copy
import cv2
class dataGenerator(object):
    def __init__(self, output_shape, meta_batch_size, task_batch_size, tfrecord_path):
        self.buffer_size=1000 # tf.data.TFRecordDataset buffer size

        self.TASK_BATCH_SIZE=task_batch_size
        self.HEIGHT, self.WIDTH, self.CHANNEL=output_shape

        self.META_BATCH_SIZE=meta_batch_size
        self.tfrecord_path = tfrecord_path
        self.label_train = self.load_tfrecord()

    def make_data_tensor(self, sess):
        #print(self.label_train.shape)
        label_train_=sess.run(self.label_train)
        input_meta =[]
        label_meta =[]
        mask_meta =[]

        for t in range(self.META_BATCH_SIZE):
            input_task = []
            label_task = []
            mask_task = []
            img=label_train_[t]
            img_patch = self.generate_patch(img)
            for idx in range(self.TASK_BATCH_SIZE*2):
                #img_HR=label_train_[t*self.TASK_BATCH_SIZE*2 + idx]
                #clean_img_LR=imresize(img_HR,scale=1./scale, kernel=Kernel)

                #img_LR=np.clip(clean_img_LR+ np.random.randn(*clean_img_LR.shape)*noise_std, 0., 1.)

                #img_ILR=imresize(img_LR, scale=scale, output_shape=img_HR.shape, kernel='cubic')
                img_label=img_patch[idx]
                img_input, img_mask = self.generate_mask(copy.deepcopy(img_label))

                input_task.append(img_input)
                label_task.append(img_label)
                mask_task.append(img_mask)

            input_meta.append(np.asarray(input_task))
            label_meta.append(np.asarray(label_task))
            mask_meta.append(np.asarray(mask_task))
        input_meta=np.asarray(input_meta)
        label_meta=np.asarray(label_meta)
        mask_meta= np.asarray(mask_meta)
        inputa=input_meta[:,:self.TASK_BATCH_SIZE,:,:]
        labela=label_meta[:,:self.TASK_BATCH_SIZE,:,:]
        maska = mask_meta[:, :self.TASK_BATCH_SIZE, :, :]
        inputb=input_meta[:,self.TASK_BATCH_SIZE:,:,:]
        labelb=label_meta[:,self.TASK_BATCH_SIZE:,:,:]
        maskb = mask_meta[:, self.TASK_BATCH_SIZE:, :, :]

        return inputa, labela, maska,inputb, labelb, maskb

    '''Load TFRECORD'''
    def _parse_function(self, example_proto):
        keys_to_features = {'label': tf.FixedLenFeature([], tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        img = parsed_features['label']
        img = tf.divide(tf.cast(tf.decode_raw(img, tf.uint8), tf.float32), 255.)
       #img = tf.reshape(img, [self.HEIGHT, self.WIDTH, self.CHANNEL])
        img = tf.reshape(img, [96, 96, self.CHANNEL])
        return img


    def load_tfrecord(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord_path)
        dataset = dataset.map(self._parse_function)

        dataset = dataset.shuffle(self.buffer_size)
        dataset = dataset.repeat()
        dataset = dataset.batch(self.META_BATCH_SIZE)
        iterator = dataset.make_one_shot_iterator()

        label_train = iterator.get_next()

        return label_train
    def generate_patch(self, img):
        h, w, c = img.shape
        patches=[]
        idy = random.sample(range(0,h-self.HEIGHT),self.TASK_BATCH_SIZE*2)
        idx = random.sample(range(0, w- self.WIDTH), self.TASK_BATCH_SIZE * 2)
        for i in range(self.TASK_BATCH_SIZE*2):
            y,x=idy[i],idx[i]
            patches.append(img[y:y+self.HEIGHT,x:x+self.WIDTH,:])
        return patches

    def generate_mask(self, input, ratio = 0.9, size_window = (5,5)):

#        ratio = self.ratio
#        size_window = self.size_window
#        size_data = self.size_data
        size_data = input.shape[:2]
        num_sample = int(size_data[0] * size_data[1] * (1 - ratio))

        mask = np.ones(size_data)
        output = input

#        for ich in range(size_data[2]):
        for ich in range(1):
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

#            id_msk = (idy_msk, idx_msk, ich)
#            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh, ich)

            id_msk = (idy_msk, idx_msk)
            id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

            output[id_msk] = input[id_msk_neigh]
            mask[id_msk] = 0.0
			 ######################################
        mask=np.stack((mask,)*3,axis=-1)
#############################################
        return output, mask
