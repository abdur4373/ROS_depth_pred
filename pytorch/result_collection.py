from predict import *

if __name__ == '__main__':
    # torch.cuda.empty_cache()

    path_to_depth = '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/nyu_depth_v2_labeled .mat'
    # read mat file
    f = h5py.File(path_to_depth)

    scenes = f['scenes']
    # scene_name_array=np.chararray([1, 1449])
    scene_name_array = []
    Delta_1_array = []
    Delta_2_array = []
    Delta_3_array = []
    abs_rel_array = []
    sqr_rel_array = []
    rmse_lin_array = []
    rmse_l_array = []
    s_no = 1

    for test_image in range(1448):  # 1448
        get_scene_name = scenes[0][test_image]
        obj = f[get_scene_name]
        scene_name = ''.join(chr(character) for character in obj[:])
        # scene_name_array.append(scene_name)
        # scene_name_array[0,j] = str1
        sequence = 'office' in scene_name

        sequence_office = 'bedroom' in scene_name

        if sequence == True:

            if sequence_office == True:
                print('skipped image {0}'.format(scene_name))

            else:

                print(s_no, scene_name, test_image)
                s_no = s_no + 1
                delta_percent_1, delta_percent_2, delta_percent_3, abs_rel, sqr_rel, rmse_lin, rmse_l = main_mod(
                    int(test_image))

                # result_images = ["test_image.jpg", "input_image.jpg", "depth_pred_inter.jpg", "sliced_depth_gt.jpg"]
                # result_col = io.imread_collection(result_images)
                # io.imshow_collection(result_col)
                # io.show()

                Delta_1_array.append(delta_percent_1)
                # print(Delta_1_array)
                Delta_2_array.append(delta_percent_2)
                Delta_3_array.append(delta_percent_3)
                abs_rel_array.append(abs_rel)
                sqr_rel_array.append(sqr_rel)
                rmse_lin_array.append(rmse_lin)
                rmse_l_array.append(rmse_l)

Delta_1_avg = sum(Delta_1_array) / len(Delta_1_array)
Delta_2_avg = sum(Delta_2_array) / len(Delta_2_array)
Delta_3_avg = sum(Delta_3_array) / len(Delta_3_array)
abs_rel_avg = sum(abs_rel_array) / len(abs_rel_array)
sqr_rel_avg = sum(sqr_rel_array) / len(sqr_rel_array)
rmse_lin_avg = sum(rmse_lin_array) / len(rmse_lin_array)
rmse_l_avg = sum(rmse_l_array) / len(rmse_l_array)

print("Delta_1_avg {0}".format(Delta_1_avg))
print("Delta_2_avg {0}".format(Delta_2_avg))
print("Delta_3_avg {0}".format(Delta_3_avg))
print("abs_rel_avg {0}".format(abs_rel_avg))
print("sqr_rel_avg {0}".format(sqr_rel_avg))
print("rmse_lin_avg {0}".format(rmse_lin_avg))
print("rmse_l_avg {0}".format(rmse_l_avg))
print("Number of images {0}".format(len(Delta_1_array)))

result_images = ["test_image.jpg", "input_image.jpg", "depth_pred_inter.jpg", "sliced_depth_gt.jpg"]
result_col = io.imread_collection(result_images)
io.imshow_collection(result_col)
io.show()
