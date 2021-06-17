import argparse
import os
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio

parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str, default='/data3/sjyang/dataset/tmp', help="dir1 to test")
parser.add_argument("--dir2", type=str, default='/data3/sjyang/dataset/tmp_result', help="dir2 to test")

args = parser.parse_args()

filenames = os.listdir(args.dir1)

psnr = 0
for filename in filenames :
    img1_path = os.path.join(args.dir1, filename)
    img2_path = os.path.join(args.dir2, filename)

    img1 = cv2.imread(img1_path).astype(np.float32) / 255.0
    img2 = cv2.imread(img2_path).astype(np.float32) / 255.0

    psnr += peak_signal_noise_ratio(img1, img2, data_range=1.0)       

print('avg psnr', psnr/len(filenames))
