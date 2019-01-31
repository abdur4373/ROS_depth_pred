import time

import scipy.misc
import torch
from PIL import Image
from comparisons import *
from depth import *
from model import *
from resize_test import *
from torch.autograd import Variable
from torchvision.utils import save_image
from utils import *
from weights import *


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

		out_img_pred_ = torch.squeeze(out_img_pred)

		out_img_pred_ = out_img_pred_.detach()

		out_img_pred_np = out_img_pred_.numpy()

		# out_norm_img = out_img_pred / depth_scale

		# print('normalized values')
		# print(out_norm_img.dtype)
		#
		#
		#
		# save_image(out_norm_img.data, "output_image.jpg")# normalize=True)

		print("Finished image in {0} s".format(time.time() - t))

		return out_img_pred_np

	def export_model(self):
		x = Variable(torch.randn(1, 3, 228, 304), requires_grad=True)
		# Export the model
		torch_out = torch.onnx._export(self.model, x, "depth_pred.onnx", export_params=True)           # model being run

# x model input (or a tuple for multiple inputs)
# where to save the model (can be a file or file-like object
# store the trained parameter weights inside the model file
#


if __name__ == '__main__':
	prediction = DepthPrediction('NYU_ResNet-UpProj.npy', 1)

	depth_gt_out = depth_gt(50)
	# img = img_as_float(imread(sys.argv[1]))
	# img=imread(sys.argv[1])
	# img=Image.open(sys.argv[1])
	img = Image.open("test2.jpg")
	out_img_pred_np = prediction.predict(img)

	# print("Predicted depth values of size 160,128 {0}".format(out_img_pred_np))
	# print('shape')
	# print(out_img_pred_np.shape)

	# print("Ground truth values of size 608,456 {0}".format(depth_gt_out))
	# print('shape')
	# print(depth_gt_out.shape)
	# print(depth_gt_out.size)

	depth_pred_inter = resize_depth_pred(out_img_pred_np)

	# print("predicted values of size 608,456 {0}".format(depth_gt_out))
	# print('shape')
	# print(depth_gt_out.shape)

	# print(list(depth_gt_out.shape))
	# print("GT Depth values {0}" .format(depth_gt_out))

	delta_calculate = delta_calculate(depth_pred_inter, depth_gt_out)

	abs_rel_diff = abs_rel_diff(depth_pred_inter, depth_gt_out)

	sqr_rel_diff = sqr_rel_diff(depth_pred_inter, depth_gt_out)

	rmse_linear = rmse_linear(depth_pred_inter, depth_gt_out)

	rmse_log = rmse_log(depth_pred_inter, depth_gt_out)

	prediction.export_model()