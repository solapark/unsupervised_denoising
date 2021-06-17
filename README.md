
## Environments
- Ubuntu 18.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6



## Guidelines for Codes

```
**Requisites should be installed beforehand.**

```
### 1. Large-Scale Pretraining

```
Please refer to the folder [**Large-Scale_Training**]

```
### 2. Meta Training

```
Download training dataset 

#### Generate TFRecord dataset
python generate_TFRecord_MZSR.py --labelpath [Path to label images] --tfrecord [TFRecord file path] --noisetype [noisetype(G,PG,MG)] --noise_a [noise_a] --noise_b [noise_b]
--if you have a problem, change the version of tensorflow 1.8 to tensorflow 1.14 only for generating TFRecord

ex) Gaussian noise
python generate_TFRecord_MZSR.py --labelpath /data3/sjyang/dataset/CBSD68 --tfrecord /data3/sjyang/MZSR+N2V/tfrecord/CBSD_gaussian_nL25 --noisetype G --noise_a 15

ex) Poisson gaussian noise
python generate_TFRecord_MZSR.py --labelpath /data3/sjyang/dataset/CBSD68 --tfrecord /data3/sjyang/MZSR+N2V/tfrecord/CBSD_poisson_nL40_10 --noisetype PG --noise_a 40 --noise_b 10

ex) Multivariate gaussian noise
python generate_TFRecord_MZSR.py --labelpath /data3/sjyang/dataset/CBSD68 --tfrecord /data3/sjyang/MZSR+N2V/tfrecord/CBSD_MG --noisetype MG

#### Train 
Make sure all configurations in **config.py** are set.

[Options]

python main.py --train --gpu [GPU_number] --trial [Trial of your training] --step [Global step]

--train: Flag in order to train.
--gpu: If you have more than one gpu in your computer, the number denotes the index. [Default 0]
--trial: Trial number. Any integer numbers can be used. [Default 0]
--step: Global step. When you resume the training, you need to specify the right global step. [Default 0]

```

### 3. Meta-Test

```
Ready for the input data 

Change model_path in main.py

[Options]

python main.py --gpu [GPU_number] --inputpath [LQ path] --gtpath [HQ path] --savepath [result path] --num [1/10]

--gpu: If you have more than one gpu in your computer, the number designates the index of GPU which is going to be used. [Default 0]
--inputpath: Path of input images [Default: Input/g20/Set5/]
--gtpath: Path of reference images. [Default: GT/Set5/]
--savepath: Path for the output images. [Default: results/Set5]
--num: [1/10] The number of adaptation (gradient updates). [Default 1]

```

