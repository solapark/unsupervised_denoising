import imageio
import os
import glob
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

from argparse import ArgumentParser

def imread(path):
    img = imageio.imread(path)
    return img

def gradients(x):
    return np.mean(((x[:-1, :-1, :] - x[1:, :-1, :]) ** 2 + (x[:-1, :-1, :] - x[:-1, 1:, :]) ** 2))

def modcrop(imgs, modulo):
    sz=imgs.shape
    sz=np.asarray(sz)
    if len(sz)==2:
        sz = sz - sz% modulo
        out = imgs[0:sz[0], 0:sz[1]]
    elif len(sz)==3:
        szt = sz[0:2]
        szt = szt - szt % modulo
        out = imgs[0:szt[0], 0:szt[1],:]

    return out

def write_to_tfrecord(writer, label, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
    }))
    writer.write(example.SerializeToString())
    return

def generate_TFRecord(label_path,tfrecord_file,patch_h,patch_w,stride):
    label_list=np.sort(np.asarray(glob.glob(label_path)))
    #img_list = np.sort(np.asarray(glob.glob(data_path)))

    offset=0

    fileNum=len(label_list)

    patches=[]
    labels=[]
    #batch_size=1
    for n in range(fileNum):
        print('[*] Image number: %d/%d' % ((n+1), fileNum))
        #img=imread(img_list[n])
        label=imread(label_list[n]) # 3D tensor [H:W:C]
        noise = torch.zeros(label.shape)
        stdN = np.random.uniform(0, 55, 1).squeeze(-1) # third parameter means batch size (num of images)
        sizeN = noise.shape
        noise = torch.FloatTensor(sizeN).normal_(mean=0,std=stdN/255.)
        img = label + noise.numpy()
        img = img.astype(np.uint8)
        print("img shape: {0}, noise.numpy() shape: {1}".format(img.shape,noise.numpy().shape))
#        assert os.path.basename(img_list[n])[:-6] == os.path.basename(label_list[n])[:-4]

#        img=modcrop(img,scale)
#        label=modcrop(label,scale)

        x, y, ch = label.shape
        for i in range(0+offset,x-patch_h+1,stride):
            for j in range(0+offset,y-patch_w+1,stride):
#                patch_d = img[i // scale:i // scale + patch_h // scale, j // scale:j // scale + patch_w // scale]
                patch_d = img[i:i + patch_h, j:j + patch_w]
                patch_l = label[i:i + patch_h, j:j + patch_w]

                if np.log(gradients(patch_l.astype(np.float64)/255.)+1e-10) >= -6.0:
                    patches.append(patch_d.tobytes())
                    labels.append(patch_l.tobytes())


    np.random.seed(36)
    np.random.shuffle(patches)
    np.random.seed(36)
    np.random.shuffle(labels)
    print('Num of patches:', len(patches))
    print('Shape: [%d, %d, %d]' % (patch_h, patch_w, ch))

    writer = tf.io.TFRecordWriter(tfrecord_file)
    for i in range(len(patches)):
        write_to_tfrecord(writer, labels[i], patches[i])

    writer.close()

if __name__=='__main__':
    parser=ArgumentParser()

    parser.add_argument('--labelpath', dest='labelpath', help='Path to HQ images ')
    #parser.add_argument('--datapath', dest='datapath', help='Path to LQ images ')
    parser.add_argument('--tfrecord', dest='tfrecord', help='Save path for tfrecord file', default='train_X1')
    options=parser.parse_args()

#    scale = options.scale
    labelpath=os.path.join(options.labelpath, '*.png')
    #datapath=os.path.join(options.datapath, '*.png')

    tfrecord_file = options.tfrecord + '.tfrecord'

    generate_TFRecord(labelpath, tfrecord_file,96,96,120)
    print('Done')

