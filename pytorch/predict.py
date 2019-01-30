
import numpy as np
import random
import os
from PIL import Image
from scipy.ndimage import imread
from skimage import img_as_float
import scipy.misc
import time
from model import *
from weights import *
from utils import *
from depth import *
import torch
from torch.autograd import Variable
from torchvision.utils import save_image
import skimage.io as io
import sys


class DepthPrediction:
	def __init__(self, weight_file, batch_size):
		self.weight_file = weight_file
		self.model = Model(batch_size)
		self.dtype = torch.cuda.FloatTensor
		self.model.load_state_dict(load_weights(self.model, self.weight_file, self.dtype))
		print("Model on cuda? {0}".format(next(self.model.parameters()).is_cuda))

	def print_model(self):
		print(self.model)

	def predict(self, img):

		# cropped_img = center_crop(img, 304, 228)
		# cropped_img=img
		resize_img = down_size(img,320,240)
		resize_img_save = torch.from_numpy(resize_img).permute(2, 0, 1).unsqueeze(0).float()
		save_image(resize_img_save, "resize_image.jpg")
		cropped_img = center_crop(resize_img, 304, 228)
		scipy.misc.toimage(cropped_img, cmin = 0.0, cmax = 1.0).save('cropped_img.jpg')
		pytorch_img = torch.from_numpy(cropped_img).permute(2,0,1).unsqueeze(0).float()
		save_image(pytorch_img, "input_image.jpg")
		pytorch_input = Variable(pytorch_img)
		print(list(pytorch_input.size()))
		t = time.time()
		out_img_pred = self.model(pytorch_input)

		# print(out_img_pred)
		# print(out_img_pred.size())
		# print('values')
		# print(out_img_pred[0][0][127][159])
		# out_norm_img = out_img_pred / 4.0
		# print('normalized values')
		# print(out_norm_img[0][0][127][159])
		# save_image(out_norm_img.data, "output_image.jpg")# normalize=True)
		print("Finished image in {0} s".format(time.time() - t))

		return out_img_pred

	def export_model(self):
		x = Variable(torch.randn(1, 3, 228, 304), requires_grad=True)
		# Export the model
		torch_out = torch.onnx._export(self.model, x, "depth_pred.onnx", export_params=True)           # model being run

# x model input (or a tuple for multiple inputs)
# where to save the model (can be a file or file-like object
# store the trained parameter weights inside the model file

	def delta_calculate(self, out_img_pred, depth_gt_out):

		for i in range(0, 160):
			for j in range(0, 128):
				print('hi')
				delta_value = np.empty([1,1,128,160])
				delta_value[0][0][i][j] = max((out_img_pred[0][0][i][j] / depth_gt_out[0][0][i][j]), (depth_gt_out[0][0][i][j] / out_img_pred[0][0][i][j]))
				j+=1
				print (delta_value[0][0][1][1])
			end
			i+=1
		end
		# Good=count(delta_value[0][0][i][j]>1.25)
		# percennt_Good=Good/20480
		# print(percennt_Good)

if __name__ == '__main__':
	prediction = DepthPrediction('NYU_ResNet-UpProj.npy', 1)
																	# img = img_as_float(imread(sys.argv[1]))
																	# img=imread(sys.argv[1])
																	# img=Image.open(sys.argv[1])
	img = Image.open("test2.jpg")
	out_img_pred=prediction.predict(img)
	depth_gt_out = depth_gt(3)

	print(list(out_img_pred.size()))
	print("Predicted depth values {0}" .format(out_img_pred))
	print(list(depth_gt_out.size()))
	print("GT Depth values {0}" .format(depth_gt_out))

	# DepthPrediction.delta_calculate(out_img_pred,depth_gt_out)
	print('export')
	prediction.export_model()