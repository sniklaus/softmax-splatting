#!/usr/bin/env python

import cv2
import glob
import numpy
import os
import skimage
import skimage.metrics
import sys
import torch

##########################################################

print('this benchmark script can be used to compute the Xiph metrics from our paper')
print('please note that it uses the SepConv method for doing the actual interpolation')
print('be aware that the script first downloads about 12 gigabytes of data from Xiph')
print('do you want to continue with the execution of this script? [y/n]')

if input().lower() != 'y':
    sys.exit(0)
# end

##########################################################

if os.path.exists('./netflix') == False:
    os.makedirs('./netflix')
# end

if len(glob.glob('./netflix/BoxingPractice-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_BoxingPractice_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/BoxingPractice-%03d.png')
# end

if len(glob.glob('./netflix/Crosswalk-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Crosswalk_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Crosswalk-%03d.png')
# end

if len(glob.glob('./netflix/DrivingPOV-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/Chimera/Netflix_DrivingPOV_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/DrivingPOV-%03d.png')
# end

if len(glob.glob('./netflix/FoodMarket-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket-%03d.png')
# end

if len(glob.glob('./netflix/FoodMarket2-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_FoodMarket2_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/FoodMarket2-%03d.png')
# end

if len(glob.glob('./netflix/RitualDance-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_RitualDance_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/RitualDance-%03d.png')
# end

if len(glob.glob('./netflix/SquareAndTimelapse-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_SquareAndTimelapse_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/SquareAndTimelapse-%03d.png')
# end

if len(glob.glob('./netflix/Tango-*.png')) != 100:
    os.system('ffmpeg -i https://media.xiph.org/video/derf/ElFuente/Netflix_Tango_4096x2160_60fps_10bit_420.y4m -pix_fmt rgb24 -vframes 100 ./netflix/Tango-%03d.png')
# end

##########################################################

if os.path.exists('./sepconv-slomo') == False:
    os.system('git clone https://github.com/sniklaus/sepconv-slomo')
    os.system('cd ./sepconv-slomo && bash download.bash')
    os.system('sed -i "s#assert(intWidth <= 1280)##g" ./sepconv-slomo/run.py')
    os.system('sed -i "s#assert(intHeight <= 720)##g" ./sepconv-slomo/run.py')
# end

sys.path.insert(0, './sepconv-slomo')
sys.path.insert(0, './sepconv-slomo/sepconv')
import run
run.arguments_strModel = 'l1'
run.arguments_strPadding = 'paper'

##########################################################

for strCategory in ['resized', 'cropped']:
    fltPsnr = []
    fltSsim = []

    for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
        for intFrame in range(2, 99, 2):
            npyFirst = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
            npySecond = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
            npyReference = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame).zfill(3) + '.png', flags=-1)

            if strCategory == 'resized':
                npyFirst = cv2.resize(src=npyFirst, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npySecond = cv2.resize(src=npySecond, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyReference = cv2.resize(src=npyReference, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

            elif strCategory == 'cropped':
                npyFirst = npyFirst[540:-540, 1024:-1024, :]
                npySecond = npySecond[540:-540, 1024:-1024, :]
                npyReference = npyReference[540:-540, 1024:-1024, :]

            # end

            tenFirst = torch.FloatTensor(numpy.ascontiguousarray(npyFirst.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            tenSecond = torch.FloatTensor(numpy.ascontiguousarray(npySecond.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            
            npyEstimate = (run.estimate(tenFirst, tenSecond).clip(0.0, 1.0).numpy().transpose(1, 2, 0) * 255.0).astype(numpy.uint8)

            fltPsnr.append(skimage.metrics.peak_signal_noise_ratio(image_true=npyReference, image_test=npyEstimate, data_range=255))
            fltSsim.append(skimage.metrics.structural_similarity(im1=npyReference, im2=npyEstimate, data_range=255, multichannel=True))
        # end
    # end

    print('category', strCategory)
    print('computed average psnr', numpy.mean(fltPsnr))
    print('computed average ssim', numpy.mean(fltSsim))
# end
