#!/usr/bin/env python

import torch

import cupy
import re

kernel_Softsplat_updateOutput = '''
	extern "C" __global__ void kernel_Softsplat_updateOutput(
		const int n,
		const float* input,
		const float* flow,
		float* output
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
		const int intC = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
		const int intY = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
		const int intX = ( intIndex                                                    ) % SIZE_3(output);

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (intSoutheastY) - dblOutputY   );
		float dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - dblOutputY   );
		float dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * (dblOutputY    - (float) (intNortheastY));
		float dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * (dblOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(output)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNorthwestY, intNorthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblNorthwest);
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(output)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intNortheastY, intNortheastX)], VALUE_4(input, intN, intC, intY, intX) * dblNortheast);
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(output)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSouthwestY, intSouthwestX)], VALUE_4(input, intN, intC, intY, intX) * dblSouthwest);
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(output)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(output))) {
			atomicAdd(&output[OFFSET_4(output, intN, intC, intSoutheastY, intSoutheastX)], VALUE_4(input, intN, intC, intY, intX) * dblSoutheast);
		}
	} }
'''

kernel_Softsplat_updateGradInput = '''
	extern "C" __global__ void kernel_Softsplat_updateGradInput(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		const int intN = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput) / SIZE_1(gradInput) ) % SIZE_0(gradInput);
		const int intC = ( intIndex / SIZE_3(gradInput) / SIZE_2(gradInput)                     ) % SIZE_1(gradInput);
		const int intY = ( intIndex / SIZE_3(gradInput)                                         ) % SIZE_2(gradInput);
		const int intX = ( intIndex                                                             ) % SIZE_3(gradInput);

		float dblGradInput = 0.0;

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (intSoutheastY) - dblOutputY   );
		float dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (intSouthwestY) - dblOutputY   );
		float dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * (dblOutputY    - (float) (intNortheastY));
		float dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * (dblOutputY    - (float) (intNorthwestY));

		if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intNorthwestY, intNorthwestX) * dblNorthwest;
		}

		if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intNortheastY, intNortheastX) * dblNortheast;
		}

		if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intSouthwestY, intSouthwestX) * dblSouthwest;
		}

		if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
			dblGradInput += VALUE_4(gradOutput, intN, intC, intSoutheastY, intSoutheastX) * dblSoutheast;
		}

		gradInput[intIndex] = dblGradInput;
	} }
'''

kernel_Softsplat_updateGradFlow = '''
	extern "C" __global__ void kernel_Softsplat_updateGradFlow(
		const int n,
		const float* input,
		const float* flow,
		const float* gradOutput,
		float* gradInput,
		float* gradFlow
	) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
		float dblGradFlow = 0.0;

		const int intN = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow) / SIZE_1(gradFlow) ) % SIZE_0(gradFlow);
		const int intC = ( intIndex / SIZE_3(gradFlow) / SIZE_2(gradFlow)                    ) % SIZE_1(gradFlow);
		const int intY = ( intIndex / SIZE_3(gradFlow)                                       ) % SIZE_2(gradFlow);
		const int intX = ( intIndex                                                          ) % SIZE_3(gradFlow);

		float dblOutputX = (float) (intX) + VALUE_4(flow, intN, 0, intY, intX);
		float dblOutputY = (float) (intY) + VALUE_4(flow, intN, 1, intY, intX);

		int intNorthwestX = (int) (floor(dblOutputX));
		int intNorthwestY = (int) (floor(dblOutputY));
		int intNortheastX = intNorthwestX + 1;
		int intNortheastY = intNorthwestY;
		int intSouthwestX = intNorthwestX;
		int intSouthwestY = intNorthwestY + 1;
		int intSoutheastX = intNorthwestX + 1;
		int intSoutheastY = intNorthwestY + 1;

		float dblNorthwest = 0.0;
		float dblNortheast = 0.0;
		float dblSouthwest = 0.0;
		float dblSoutheast = 0.0;

		if (intC == 0) {
			dblNorthwest = ((float) (-1.0)) * ((float) (intSoutheastY) - dblOutputY   );
			dblNortheast = ((float) (+1.0)) * ((float) (intSouthwestY) - dblOutputY   );
			dblSouthwest = ((float) (-1.0)) * (dblOutputY    - (float) (intNortheastY));
			dblSoutheast = ((float) (+1.0)) * (dblOutputY    - (float) (intNorthwestY));

		} else if (intC == 1) {
			dblNorthwest = ((float) (intSoutheastX) - dblOutputX   ) * ((float) (-1.0));
			dblNortheast = (dblOutputX    - (float) (intSouthwestX)) * ((float) (-1.0));
			dblSouthwest = ((float) (intNortheastX) - dblOutputX   ) * ((float) (+1.0));
			dblSoutheast = (dblOutputX    - (float) (intNorthwestX)) * ((float) (+1.0));

		}

		for (int intChannel = 0; intChannel < SIZE_1(gradOutput); intChannel += 1) {
			float dblInput = VALUE_4(input, intN, intChannel, intY, intX);

			if ((intNorthwestX >= 0) & (intNorthwestX < SIZE_3(gradOutput)) & (intNorthwestY >= 0) & (intNorthwestY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intNorthwestY, intNorthwestX) * dblNorthwest;
			}

			if ((intNortheastX >= 0) & (intNortheastX < SIZE_3(gradOutput)) & (intNortheastY >= 0) & (intNortheastY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intNortheastY, intNortheastX) * dblNortheast;
			}

			if ((intSouthwestX >= 0) & (intSouthwestX < SIZE_3(gradOutput)) & (intSouthwestY >= 0) & (intSouthwestY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intSouthwestY, intSouthwestX) * dblSouthwest;
			}

			if ((intSoutheastX >= 0) & (intSoutheastX < SIZE_3(gradOutput)) & (intSoutheastY >= 0) & (intSoutheastY < SIZE_2(gradOutput))) {
				dblGradFlow += dblInput * VALUE_4(gradOutput, intN, intChannel, intSoutheastY, intSoutheastX) * dblSoutheast;
			}
		}

		gradFlow[intIndex] = dblGradFlow;
	} }
'''

def cupy_kernel(strFunction, objectVariables):
	strKernel = globals()[strFunction]

	while True:
		objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArg = int(objectMatch.group(2))

		strTensor = objectMatch.group(4)
		intSizes = objectVariables[strTensor].size()

		strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
	# end

	while True:
		objectMatch = re.search('(OFFSET_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), '(' + str.join('+', strIndex) + ')')
	# end

	while True:
		objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

		if objectMatch is None:
			break
		# end

		intArgs = int(objectMatch.group(2))
		strArgs = objectMatch.group(4).split(',')

		strTensor = strArgs[0]
		intStrides = objectVariables[strTensor].stride()
		strIndex = [ '((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs) ]

		strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
	# end

	return strKernel
# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
	return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
# end

class _FunctionSoftsplat(torch.autograd.Function):
	@staticmethod
	def forward(self, input, flow):
		self.save_for_backward(input, flow)

		intSamples = input.shape[0]
		intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
		intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

		assert(intFlowDepth == 2)
		assert(intInputHeight == intFlowHeight)
		assert(intInputWidth == intFlowWidth)

		assert(input.is_contiguous() == True)
		assert(flow.is_contiguous() == True)

		output = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ])

		if input.is_cuda == True:
			n = output.nelement()
			cupy_launch('kernel_Softsplat_updateOutput', cupy_kernel('kernel_Softsplat_updateOutput', {
				'input': input,
				'flow': flow,
				'output': output
			}))(
				grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
				block=tuple([ 512, 1, 1 ]),
				args=[ n, input.data_ptr(), flow.data_ptr(), output.data_ptr() ]
			)

		elif first.is_cuda == False:
			raise NotImplementedError()

		# end

		return output
	# end

	@staticmethod
	def backward(self, gradOutput):
		input, flow = self.saved_tensors

		intSamples = input.shape[0]
		intInputDepth, intInputHeight, intInputWidth = input.shape[1], input.shape[2], input.shape[3]
		intFlowDepth, intFlowHeight, intFlowWidth = flow.shape[1], flow.shape[2], flow.shape[3]

		assert(intFlowDepth == 2)
		assert(intInputHeight == intFlowHeight)
		assert(intInputWidth == intFlowWidth)

		assert(gradOutput.is_contiguous() == True)

		gradInput = input.new_zeros([ intSamples, intInputDepth, intInputHeight, intInputWidth ]) if self.needs_input_grad[0] == True else None
		gradFlow = input.new_zeros([ intSamples, intFlowDepth, intFlowHeight, intFlowWidth ]) if self.needs_input_grad[1] == True else None

		if input.is_cuda == True:
			if gradInput is not None:
				n = gradInput.nelement()
				cupy_launch('kernel_Softsplat_updateGradInput', cupy_kernel('kernel_Softsplat_updateGradInput', {
					'input': input,
					'flow': flow,
					'gradOutput': gradOutput,
					'gradInput': gradInput,
					'gradFlow': gradFlow
				}))(
					grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), gradInput.data_ptr(), None ]
				)
			# end

			if gradFlow is not None:
				n = gradFlow.nelement()
				cupy_launch('kernel_Softsplat_updateGradFlow', cupy_kernel('kernel_Softsplat_updateGradFlow', {
					'input': input,
					'flow': flow,
					'gradOutput': gradOutput,
					'gradInput': gradInput,
					'gradFlow': gradFlow
				}))(
					grid=tuple([ int((n + 512 - 1) / 512), 1, 1 ]),
					block=tuple([ 512, 1, 1 ]),
					args=[ n, input.data_ptr(), flow.data_ptr(), gradOutput.data_ptr(), None, gradFlow.data_ptr() ]
				)
			# end

		elif input.is_cuda == False:
			raise NotImplementedError()

		# end

		return gradInput, gradFlow
	# end
# end

def FunctionSoftsplat(tensorInput, tensorFlow, tensorMetric, strType):
	assert(tensorMetric is None or tensorMetric.shape[1] == 1)
	assert(strType in ['summation', 'average', 'linear', 'softmax'])

	if strType == 'average':
		tensorInput = torch.cat([ tensorInput, tensorInput.new_ones(tensorInput.shape[0], 1, tensorInput.shape[2], tensorInput.shape[3]) ], 1)

	elif strType == 'linear':
		tensorInput = torch.cat([ tensorInput * tensorMetric, tensorMetric ], 1)

	elif strType == 'softmax':
		tensorInput = torch.cat([ tensorInput * tensorMetric.exp(), tensorMetric.exp() ], 1)

	# end

	tensorOutput = _FunctionSoftsplat.apply(tensorInput, tensorFlow)

	if strType != 'summation':
		tensorOutput = tensorOutput[:, :-1, :, :] / (tensorOutput[:, -1:, :, :] + 0.0000001)
	# end

	return tensorOutput
# end

class ModuleSoftsplat(torch.nn.Module):
	def __init__(self, strType):
		super(ModuleSoftsplat, self).__init__()

		self.strType = strType
	# end

	def forward(self, tensorInput, tensorFlow, tensorMetric):
		return FunctionSoftsplat(tensorInput, tensorFlow, tensorMetric, strType)
	# end
# end