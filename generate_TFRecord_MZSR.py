import imageio
import os
import glob
import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from scipy.linalg import orth

import cv2

def augmentation(x,mode):
    if mode ==0:
        y=x

    elif mode ==1:
        y=np.flipud(x)

    elif mode == 2:
        y = np.rot90(x,1)

    elif mode == 3:
        y = np.rot90(x, 1)
        y = np.flipud(y)

    elif mode == 4:
        y = np.rot90(x, 2)

    elif mode == 5:
        y = np.rot90(x, 2)
        y = np.flipud(y)

    elif mode == 6:
        y = np.rot90(x, 3)

    elif mode == 7:
        y = np.rot90(x, 3)
        y = np.flipud(y)

    return y

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def write_to_tfrecord(writer, label):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(noise_type,noise_level,label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))

    offset=0

    fileNum=len(label_list)

    labels=[]
    LQs=[]
    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        label=imread(label_list[n])
        LQ = np.zeros(label.shape,np.uint8)
        if(noise_type == 'G'):
            noise = np.random.normal(0, noise_level[0]/1., label.shape).astype(np.uint8)
            LQ = label + noise
            imageio.imwrite('/data3/sjyang/MZSR+N2V/tfrecord/gaussian_nL{0}_img/LQ{1}.png'.format(noise_level[0],n),LQ)
        elif(noise_type == 'PG'):
            sigma_s = noise_level[0]/255.
            sigma_c = noise_level[1]/255.
            n1 = np.random.randn(*label.shape)*sigma_s*(label/255.)
            n2 = np.random.randn(*label.shape)*sigma_c
            noise = ((n1+n2)*255.).astype(np.uint8)
            LQ = label + noise
            imageio.imwrite('/data3/sjyang/MZSR+N2V/tfrecord/poison_nL{0}_{1}_img/LQ{2}.png'.format(noise_level[0],noise_level[1],n),LQ)
        elif(noise_type =='MG'):
            H,W,_ = label.shape
            L = 75/255.
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3,3))
            tmp = np.matmul(D,U)
            tmp = np.matmul(U.T,np.matmul(D,U))
            tmp = (L**2)*tmp
            noiseSigma = np.abs(tmp)
            noise = np.random.multivariate_normal([0,0,0], noiseSigma, (H,W)).astype(np.uint8)
            LQ = label + (noise*255)
            imageio.imwrite('/data3/sjyang/MZSR+N2V/tfrecord/MG_img_revised/LQ{}.png'.format(n),LQ)
        #imageio.imwrite('/data3/sjyang/MZSR+N2V/gaussian_nL25_img/HQ{0}_{1}_nL{2}.png'.format(n,noise_type,noise_level[0]),label)
        x, y, ch = label.shape
        count=0
        for m in range(8):
            for i in range(0+offset,x-patch_h+1,stride):
                for j in range(0+offset,y-patch_w+1,stride):
                    patch_l = label[i:i + patch_h, j:j + patch_w]
                    patch_lq = LQ[i:i + patch_h, j:j + patch_w]
                    if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                        #labels.append(augmentation(patch_l,m).tobytes())
                        LQs.append(augmentation(patch_lq,m).tobytes())
                        count=count+1
                        #cv2.imwrite('/data3/sjyang/label{}.png'.format(count),patch_l)
                        #cv2.imwrite('/data3/sjyang/LQ{}.png'.format(count),patch_lq)
    np.random.shuffle(LQs)
    print('Num of patches:', len(LQs))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(LQs)):
        if i % 10000 == 0:
            print('[%d/%d] Processed' % ((i+1), len(LQs)))
        write_to_tfrecord(writer, LQs[i])

    writer.close()

if __name__=='__main__':
    parser=ArgumentParser()
    parser.add_argument('--labelpath', dest='labelpath', help='Path to HR images (./DIV2K_train_HR)')
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='train_SR_MZSR')
    
    parser.add_argument('--noisetype', dest='noisetype', choices=['G','PG','MG'], help='Noise type synthesized to HQ for generating synthetic LQ')
    parser.add_argument('--noise_a', dest='noise_a', type=int, help='std for G, alpha for PG', default=15)
    parser.add_argument('--noise_b', dest='noise_b', type=int, help='delta for PG', default=10)

    options=parser.parse_args()

    labelpath=os.path.join(options.labelpath, '*.png')
    tfrecord_file = options.tfrecord + '.tfrecord'
    noise_type = options.noisetype
    noise_level = [options.noise_a, options.noise_b]
    generate_TFRecord(noise_type, noise_level, labelpath, tfrecord_file,96,96,120)
    print('Done')

