#!/usr/bin/env python

import glob
import numpy
import PIL
import PIL.Image
import skimage
import skimage.metrics
import torch

import run

##########################################################

run.args_strModel = 'l1'

##########################################################

if __name__ == '__main__':
    fltPsnr = []
    fltSsim = []

    for strTruth in sorted(glob.glob('./middlebury/*/frame10i11.png')):
        tenOne = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(strTruth.replace('frame10i11', 'frame10')))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
        tenTwo = torch.FloatTensor(numpy.ascontiguousarray(numpy.array(PIL.Image.open(strTruth.replace('frame10i11', 'frame11')))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

        npyEstimate = (run.estimate(tenOne, tenTwo, [0.5])[0].clip(0.0, 1.0).numpy(force=True).transpose(1, 2, 0) * 255.0).round().astype(numpy.uint8)

        fltPsnr.append(skimage.metrics.peak_signal_noise_ratio(image_true=numpy.array(PIL.Image.open(strTruth))[:, :, ::-1], image_test=npyEstimate, data_range=255))
        fltSsim.append(skimage.metrics.structural_similarity(im1=numpy.array(PIL.Image.open(strTruth))[:, :, ::-1], im2=npyEstimate, data_range=255, channel_axis=2))
    # end

    print('computed average psnr', numpy.mean(fltPsnr))
    print('computed average ssim', numpy.mean(fltSsim))
# end