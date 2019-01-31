# from predict import*
import numpy as np


# Good=count(delta_value[0][0][i][j]>1.25)
# percennt_Good=Good/20480
# print(percennt_Good)


def delta_calculate(depth_pred_inter, depth_gt_out):
    print('depth_pred_inter')
    print(depth_pred_inter, depth_pred_inter.shape)

    print('depth_gt_out')
    print(depth_gt_out, depth_gt_out.shape)

    pred_div_gt = np.true_divide(depth_pred_inter, depth_gt_out)
    gt_div_pred = np.true_divide(depth_gt_out, depth_pred_inter)
    delta = np.maximum(pred_div_gt, gt_div_pred) < 1.25

    print('maximum value')
    print(delta.size)

    total_no_of_delta = np.sum(delta)

    print('total_max_value less than 1.25')
    print(total_no_of_delta)

    # print('lenght_total_max_value')
    # print(lenght_total_max_value, total_max_value.shape)

    delta_percent = total_no_of_delta * 100 / delta.size

    # pred_div_gt_max = pred_div_gt > 1.25
    #
    # gt_div_pred_max=gt_div_pred > 1.25

    print('Delta 1 %')
    print(delta_percent)
    # len(pred_div_gt[pred_div_gt_max])
    # print(len(pred_div_gt[pred_div_gt_max]))
    #
    # print('gt_div_pred values greater than 1.25')
    # print(gt_div_pred_max)
    # len(gt_div_pred[gt_div_pred_max])
    # print(len(gt_div_pred[gt_div_pred_max]))

    # for i in range(0, 608):


# 	for j in range(0, 456):
# 		print('hi')
# 		delta_value = np.empty([456,608])
# 		delta_value [i,j]= max((depth_pred_inter[i, j] / depth_gt_out[i, j], (depth_gt_out[i, j] / depth_pred_inter[i, j])
# 		# print(delta_value)
#


# x = np.array([[2,2,2],[4,4,4],[3,3,3]])
# z = np.array([[1,1,1],[4,4,4],[3,3,3]])
# y=np.true_divide(x, z)
#
# print(x>1)
# print(y)

# for i in range(3):
#     for j in range(3):
#     q=x[i][j]-z[i][j]

# y = np.array([[123,24123,32432], [234,24,23]])
# print(y)
#
# b = y > 200
#
# print(b)
#
# print(len(y[b]))
#
# print(y[b].sum())

def abs_rel_diff(depth_pred_inter, depth_gt_out):
    abs_diff_pixel_wise = np.true_divide(abs(np.subtract(depth_pred_inter, depth_gt_out)), depth_gt_out)
    num_of_pixels = depth_gt_out.size
    abs_diff = np.sum(abs_diff_pixel_wise) / num_of_pixels

    print("number of pixels {0}".format(num_of_pixels))
    print("Absolute relative difference {0}".format(abs_diff))


def sqr_rel_diff(depth_pred_inter, depth_gt_out):
    sqr_diff_pixel_wise = np.true_divide(np.square(np.subtract(depth_pred_inter, depth_gt_out)), depth_gt_out)
    num_of_pixels = depth_gt_out.size
    sqr_diff = np.sum(sqr_diff_pixel_wise) / num_of_pixels

    print("Squared relative difference {0}".format(sqr_diff))


def rmse_linear(depth_pred_inter, depth_gt_out):
    rmse_diff_pixel_wise = np.square(np.subtract(depth_pred_inter, depth_gt_out))
    num_of_pixels = depth_gt_out.size
    rmse_linear = np.sqrt(np.sum(rmse_diff_pixel_wise) / num_of_pixels)

    print("RMSE (Linear) {0}".format(rmse_linear))


def rmse_log(depth_pred_inter, depth_gt_out):
    rmse_log_diff_pixel_wise = np.square(np.subtract(np.log10(depth_pred_inter), np.log10(depth_gt_out)))
    num_of_pixels = depth_gt_out.size
    rmse_log = np.sqrt(np.sum(rmse_log_diff_pixel_wise) / num_of_pixels)

    print("RMSE (Log) {0}".format(rmse_log))
