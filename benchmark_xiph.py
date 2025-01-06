#!/usr/bin/env python

import cv2
import glob
import numpy
import os
import skimage
import skimage.metrics
import sys
import torch

import run

##########################################################

run.args_strModel = 'l1'

##########################################################

os.makedirs(name='./netflix', exist_ok=True)

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

for strCategory in ['resized', 'cropped']:
    fltPsnr = []
    fltSsim = []

    for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
        for intFrame in range(2, 99, 2):
            npyOne = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
            npyTwo = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
            npyTruth = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame).zfill(3) + '.png', flags=-1)

            if strCategory == 'resized':
                npyOne = cv2.resize(src=npyOne, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyTwo = cv2.resize(src=npyTwo, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
                npyTruth = cv2.resize(src=npyTruth, dsize=(2048, 1080), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

            elif strCategory == 'cropped':
                npyOne = npyOne[540:-540, 1024:-1024, :]
                npyTwo = npyTwo[540:-540, 1024:-1024, :]
                npyTruth = npyTruth[540:-540, 1024:-1024, :]

            # end

            tenOne = torch.FloatTensor(numpy.ascontiguousarray(npyOne.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
            tenTwo = torch.FloatTensor(numpy.ascontiguousarray(npyTwo.transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

            npyEstimate = (run.estimate(tenOne, tenTwo, [0.5])[0].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0) * 255.0).round().astype(numpy.uint8)

            fltPsnr.append(skimage.metrics.peak_signal_noise_ratio(image_true=npyTruth, image_test=npyEstimate, data_range=255))
            fltSsim.append(skimage.metrics.structural_similarity(im1=npyTruth, im2=npyEstimate, data_range=255, channel_axis=2))
        # end
    # end

    print('category', strCategory)
    print('computed average psnr', numpy.mean(fltPsnr))
    print('computed average ssim', numpy.mean(fltSsim))
# end
