import h5py

path_to_depth = '/media/ghost/8c5dd22b-3c0c-41d2-9807-c4094164ca3e/ghost/down/nyu_depth_v2_labeled .mat'
f = h5py.File(path_to_depth, 'r')

for key in f.keys():
    print(key)
