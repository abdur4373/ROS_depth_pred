import h5py

path_to_depth = '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/nyu_depth_v2_labeled .mat'
# read mat file
f = h5py.File(path_to_depth)

# read 0-th image. original format is [3 x 640 x 480], uint8
# img = f['images']
scenes = f['scenes']
type_scene = f['sceneTypes']
# scene_name_array=np.chararray([1, 1449])
scene_name_array = []
scene_type_array = []
for image_no in range(1448):

    scene_name = scenes[0][image_no]
    scene_type = type_scene[0][image_no]
    obj1 = f[scene_name]
    obj2 = f[scene_type]
    str1 = ''.join(chr(i) for i in obj1[:])
    str2 = ''.join(chr(j) for j in obj2[:])

    # scene_name_array.append(str1)
    # scene_type_array.append(str2)

    s = 'bedroom' in str1

    if s == True:
        print(str1, str2, image_no)
        scene_name_array.append(str1)

    # print(img)

# print(Counter(scene_name_array))
print(len(scene_name_array))
