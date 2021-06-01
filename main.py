import dataGenerator
import train
import test
from utils import *
from config import *
import glob
import scipy.io

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

conf=tf.ConfigProto()
conf.gpu_options.per_process_gpu_memory_fraction=0.95

def main():
    if args.is_train==True:
        data_generator=dataGenerator.dataGenerator(output_shape=[HEIGHT,WIDTH,CHANNEL], meta_batch_size=META_BATCH_SIZE,
                                                   task_batch_size=TASK_BATCH_SIZE,tfrecord_path=TFRECORD_PATH)

        Trainer = train.Train(trial=args.trial, step=args.step, size=[HEIGHT,WIDTH,CHANNEL],
                              meta_batch_size=META_BATCH_SIZE, meta_lr=META_LR, meta_iter=META_ITER, task_batch_size=TASK_BATCH_SIZE,
                              task_lr=TASK_LR, task_iter=TASK_ITER, data_generator=data_generator, checkpoint_dir=CHECKPOINT_DIR, conf=conf)

        Trainer()
    else:
        model_path='SR-gray/Model0/model-50000'

        img_path=sorted(glob.glob(os.path.join(args.inputpath, '*.png')))
        gt_path=sorted(glob.glob(os.path.join(args.gtpath, '*.png')))


        Tester=test.Test(model_path, args.savepath, conf, args.num_of_adaptation)
        P=[]
        for i in range(len(img_path)):
            img=imread(img_path[i])
            gt=imread(gt_path[i])

            _, pp =Tester(img, gt, img_path[i])

            P.append(pp)

        avg_PSNR=np.mean(P, 0)

        print('[*] Average PSNR ** Initial: %.4f, Final : %.4f' % tuple(avg_PSNR))


if __name__=='__main__':
    main()
