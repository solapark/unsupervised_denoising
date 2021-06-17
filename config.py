from argparse import ArgumentParser

parser=ArgumentParser()

# Global
parser.add_argument('--gpu', type=str, dest='gpu', default='0')

# For Meta-test
parser.add_argument('--inputpath', type=str, dest='inputpath', default='TestSet/Set5/g13/LR/')
parser.add_argument('--gtpath', type=str, dest='gtpath', default='TestSet/Set5/GT_crop/')
#parser.add_argument('--kernelpath', type=str, dest='kernelpath', default='TestSet/Set5/g13/kernel.mat')
parser.add_argument('--savepath', type=str, dest='savepath', default='results/Set5')
#parser.add_argument('--model', type=int, dest='model', choices=[0,1,2,3], default=0)
parser.add_argument('--num', type=int, dest='num_of_adaptation', choices=[1,10], default=1)

# For Meta-Training
parser.add_argument('--trial', type=int, dest='trial', default=0)
parser.add_argument('--step', type=int, dest='step', default=0)
parser.add_argument('--train', dest='is_train', default=False, action='store_true')

args= parser.parse_args()

#Transfer Learning From Pre-trained model.
IS_TRANSFER = True
TRANS_MODEL = '/data3/sap/unsupervised_denoising/checkpoint/Large-Scale_Training/Model0/model-800000'

# Dataset Options
HEIGHT=64
WIDTH=64
CHANNEL=3

# SCALE_LIST=[2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
SCALE_LIST=[2.0]

META_ITER=100000
META_BATCH_SIZE=5
META_LR=1e-4

TASK_ITER=5
TASK_BATCH_SIZE=2
TASK_LR=1e-2

# Loading tfrecord and saving paths
TFRECORD_PATH='/data3/sjyang/MZSR+N2V/tfrecord/CBSD_gaussian_nL50.tfrecord'
CHECKPOINT_DIR='/data3/sjyang/MZSR+N2V/checkpoint/CBSD_gaussian_nL50'
