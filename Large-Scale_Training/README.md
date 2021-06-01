# Large-Scale Training Codes

## Environments
- Ubuntu 18.04
- [Tensorflow 1.8](http://www.tensorflow.org/)
- CUDA 9.0 & cuDNN 7.1
- Python 3.6

## Guidelines for Codes

**Requisites should be installed beforehand.**

### Training

```
Download training dataset 

#### Generate TFRecord dataset
python generate_TFRecord.py --labelpath [Path to HQ images] --datapath [Path to LQ images] --tfrecord [TFRecord file path]
--if you have a problem, change the version of tensorflow 1.8 to tensorflow 1.14 only for generating TFRecord

#### Train

-- change TF_RECORD_PATH in main.py

[Options]

python main.py --gpu [GPU_number] --trial [Trial of your training] --step [Global step]

--gpu: If you have more than one gpu in your computer, the number denotes the index. [Default 0]
--trial: Trial number. Any integer numbers can be used. [Default 0]
--step: Global step. When you resume the training, you need to specify the right global step. [Default 0]

** An Example Code **

python main.py --gpu 0 --trial 1 --step 0


```
