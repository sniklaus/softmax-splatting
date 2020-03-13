#!/usr/bin/env python

import torch

import cv2
import numpy

import softsplat

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################

backwarp_tensorGrid = {}

def backwarp(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in backwarp_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.shape[3]).view(1, 1, 1, tensorFlow.shape[3]).expand(tensorFlow.shape[0], -1, tensorFlow.shape[2], -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.shape[2]).view(1, 1, tensorFlow.shape[2], 1).expand(tensorFlow.shape[0], -1, -1, tensorFlow.shape[3])

		backwarp_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()
	# end

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.shape[3] - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(backwarp_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
# end

##########################################################

numpyFirst = cv2.imread(filename='./images/first.png', flags=-1).astype(numpy.float32) / 255.0
numpySecond = cv2.imread(filename='./images/second.png', flags=-1).astype(numpy.float32) / 255.0
numpyFlow = cv2.optflow.readOpticalFlow(path='./images/flow.flo')

tensorFirst = torch.FloatTensor(numpyFirst.transpose(2, 0, 1)[None, :, :, :]).cuda()
tensorSecond = torch.FloatTensor(numpySecond.transpose(2, 0, 1)[None, :, :, :]).cuda()
tensorFlow = torch.FloatTensor(numpyFlow.transpose(2, 0, 1)[None, :, :, :]).cuda()

tensorMetric = torch.nn.functional.l1_loss(input=tensorFirst, target=backwarp(tensorInput=tensorSecond, tensorFlow=tensorFlow), reduction='none').mean(1, True)

for intTime, dblTime in enumerate(numpy.linspace(0.0, 1.0, 11).tolist()):
	tensorSummation = softsplat.FunctionSoftsplat(tensorInput=tensorFirst, tensorFlow=tensorFlow * dblTime, tensorMetric=None, strType='summation')
	tensorAverage = softsplat.FunctionSoftsplat(tensorInput=tensorFirst, tensorFlow=tensorFlow * dblTime, tensorMetric=None, strType='average')
	tensorLinear = softsplat.FunctionSoftsplat(tensorInput=tensorFirst, tensorFlow=tensorFlow * dblTime, tensorMetric=(0.3 - tensorMetric).clamp(0.0000001, 1.0), strType='linear') # finding a good linearly metric is difficult, and it is not invariant to translations
	tensorSoftmax = softsplat.FunctionSoftsplat(tensorInput=tensorFirst, tensorFlow=tensorFlow * dblTime, tensorMetric=-20.0 * tensorMetric, strType='softmax') # -20.0 is a hyperparameter, called 'beta' in the paper, that could be learned using a torch.Parameter

	cv2.imshow(winname='summation', mat=tensorSummation[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='average', mat=tensorAverage[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='linear', mat=tensorLinear[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='softmax', mat=tensorSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.waitKey(delay=0)
# end