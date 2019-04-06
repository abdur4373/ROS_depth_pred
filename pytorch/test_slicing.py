# from predict import*
import numpy as np
import numpy.ma as ma

def delta_calculate(depth_pred_inter, depth_gt_out):


    pred_div_gt = np.true_divide(depth_pred_inter, depth_gt_out)
    pred_div_gt_masked = ma.masked_invalid(pred_div_gt)
    gt_div_pred = np.true_divide(depth_gt_out, depth_pred_inter)
    gt_div_pred_masked = ma.masked_equal(gt_div_pred, 0)
    delta_1 = np.maximum(pred_div_gt_masked, gt_div_pred_masked) < 1.25
    delta_2 = np.maximum(pred_div_gt_masked, gt_div_pred_masked) < np.square(1.25)
    delta_3 = np.maximum(pred_div_gt_masked, gt_div_pred_masked) < np.power(1.25, 3)

    # pred_div_gt = np.true_divide(depth_pred_inter, depth_gt_out)
    # gt_div_pred = np.true_divide(depth_gt_out, depth_pred_inter)
    # delta_1 = np.maximum(pred_div_gt, gt_div_pred) < 1.25
    # delta_2 = np.maximum(pred_div_gt, gt_div_pred) < np.square(1.25)
    # delta_3 = np.maximum(pred_div_gt, gt_div_pred) < np.power(1.25, 3)
    # print('maximum value')
    # print(delta.size)

    total_no_of_delta_1 = np.sum(delta_1)
    total_no_of_delta_2 = np.sum(delta_2)
    total_no_of_delta_3 = np.sum(delta_3)
    # print('total_max_value less than 1.25')
    # print(total_no_of_delta)

    # print('lenght_total_max_value')
    # print(lenght_total_max_value, total_max_value.shape)
    delta_percent_1 = total_no_of_delta_1 * 100 / delta_1.count()
    delta_percent_2 = total_no_of_delta_2 * 100 / delta_2.count()
    delta_percent_3 = total_no_of_delta_3 * 100 / delta_3.count()

    # delta_percent_1 = total_no_of_delta_1 * 100 / delta_1.size
    # delta_percent_2 = total_no_of_delta_2 * 100 / delta_2.size
    # delta_percent_3 = total_no_of_delta_3 * 100 / delta_3.size
    # print("Delta_1: {0}".format(delta_percent_1),"%")
    # print("Delta_2: {0}".format(delta_percent_2), "%")
    # print("Delta_3: {0}".format(delta_percent_3), "%")

    return delta_percent_1, delta_percent_2, delta_percent_3

    # len(pred_div_gt[pred_div_gt_max])
    # print(len(pred_div_gt[pred_div_gt_max]))
    #
    # print('gt_div_pred values greater than 1.25')
    # print(gt_div_pred_max)
    # len(gt_div_pred[gt_div_pred_max])
    # print(len(gt_div_pred[gt_div_pred_max]))

    # for i in range(0, 608):


def abs_rel_diff(depth_pred_inter, depth_gt_out):
    abs_diff_pixel_wise = np.true_divide(abs(np.subtract(depth_pred_inter, depth_gt_out)), depth_gt_out)
    abs_diff_pixel_wise_masked = ma.masked_invalid(abs_diff_pixel_wise)
    num_of_pixels = abs_diff_pixel_wise_masked.count()
    abs_diff = np.sum(abs_diff_pixel_wise_masked) / num_of_pixels

    # print("number of pixels {0}".format(num_of_pixels))
    # print("Absolute relative difference {0}".format(abs_diff))
    return abs_diff


def sqr_rel_diff(depth_pred_inter, depth_gt_out):
    sqr_diff_pixel_wise = np.true_divide(np.square(np.subtract(depth_pred_inter, depth_gt_out)), depth_gt_out)
    abs_diff_pixel_wise_masked = ma.masked_invalid(sqr_diff_pixel_wise)
    num_of_pixels = abs_diff_pixel_wise_masked.count()
    sqr_diff = np.sum(abs_diff_pixel_wise_masked) / num_of_pixels

    # print("Squared relative difference {0}".format(sqr_diff))
    return sqr_diff


def rmse_linear(depth_pred_inter, depth_gt_out):
    rmse_diff_pixel_wise = np.square(np.subtract(depth_pred_inter, depth_gt_out))
    num_of_pixels = depth_gt_out.size
    rmse_linear = np.sqrt(np.sum(rmse_diff_pixel_wise) / num_of_pixels)

    # print("RMSE (Linear) {0}".format(rmse_linear))
    return rmse_linear


def rmse_log(depth_pred_inter, depth_gt_out):
    rmse_log_diff_pixel_wise = np.square(np.subtract(np.log10(depth_pred_inter), np.log10(depth_gt_out)))
    num_of_pixels = depth_gt_out.size
    rmse_log = np.sqrt(np.sum(rmse_log_diff_pixel_wise) / num_of_pixels)

    # print("RMSE (Log) {0}".format(rmse_log))
    return rmse_log
