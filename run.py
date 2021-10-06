#!/usr/bin/env python

import torch

import cv2
import numpy

import softsplat

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0

##########################################################

def read_flo(strFile):
    with open(strFile, 'rb') as objFile:
        strFlow = objFile.read()
    # end

    assert(numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=1, offset=0) == 202021.25)

    intWidth = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=4)[0]
    intHeight = numpy.frombuffer(buffer=strFlow, dtype=numpy.int32, count=1, offset=8)[0]

    return numpy.frombuffer(buffer=strFlow, dtype=numpy.float32, count=intHeight * intWidth * 2, offset=12).reshape(intHeight, intWidth, 2)
# end

##########################################################

backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
	if str(tenFlow.shape) not in backwarp_tenGrid:
		tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
		tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

		backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
	# end

	tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

##########################################################

tenOne = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/one.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
tenTwo = torch.FloatTensor(numpy.ascontiguousarray(cv2.imread(filename='./images/two.png', flags=-1).transpose(2, 0, 1)[None, :, :, :].astype(numpy.float32) * (1.0 / 255.0))).cuda()
tenFlow = torch.FloatTensor(numpy.ascontiguousarray(read_flo('./images/flow.flo').transpose(2, 0, 1)[None, :, :, :])).cuda()

tenMetric = torch.nn.functional.l1_loss(input=tenOne, target=backwarp(tenInput=tenTwo, tenFlow=tenFlow), reduction='none').mean(1, True)

for intTime, fltTime in enumerate(numpy.linspace(0.0, 1.0, 11).tolist()):
	tenSummation = softsplat.FunctionSoftsplat(tenInput=tenOne, tenFlow=tenFlow * fltTime, tenMetric=None, strType='summation')
	tenAverage = softsplat.FunctionSoftsplat(tenInput=tenOne, tenFlow=tenFlow * fltTime, tenMetric=None, strType='average')
	tenLinear = softsplat.FunctionSoftsplat(tenInput=tenOne, tenFlow=tenFlow * fltTime, tenMetric=(0.3 - tenMetric).clip(0.0000001, 1.0), strType='linear') # finding a good linearly metric is difficult, and it is not invariant to translations
	tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenOne, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric, strType='softmax') # -20.0 is a hyperparameter, called 'alpha' in the paper, that could be learned using a torch.Parameter

	cv2.imshow(winname='summation', mat=tenSummation[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='average', mat=tenAverage[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='linear', mat=tenLinear[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.imshow(winname='softmax', mat=tenSoftmax[0, :, :, :].cpu().numpy().transpose(1, 2, 0))
	cv2.waitKey(delay=0)
# end