import numpy as np
import skimage.io as io
# from depth import*
from predict import *
from scipy import interpolate


def resize_depth_pred(out_img_pred_np):
    x = [x for x in range(160)]
    y = [y for y in range(128)]

    f = interpolate.interp2d(x, y, out_img_pred_np)

    xnew = np.linspace(0, 160, 608)

    ynew = np.linspace(0, 128, 456)

    depth_pred_inter = f(xnew, ynew)

    # print("Predicted depth values of size 608,456 {0}".format(depth_pred_inter))
    # print('shape')
    # print(depth_pred_inter.shape)
    depth_pred_inter_diff = depth_pred_inter - np.amin(depth_pred_inter)
    depth_pred_inter_norm = depth_pred_inter_diff / np.amax(depth_pred_inter_diff)
    depth_pred_inter_ = np.empty([456, 608, 3])
    depth_pred_inter_[:, :, 0] = depth_pred_inter_norm[:, :]
    depth_pred_inter_[:, :, 1] = depth_pred_inter_norm[:, :]
    depth_pred_inter_[:, :, 2] = depth_pred_inter_norm[:, :]
    # io.imshow(depth_pred_inter_ / 4.0)
    io.imsave('depth_pred_inter.jpg', depth_pred_inter_)

    # io.show()

    return depth_pred_inter
