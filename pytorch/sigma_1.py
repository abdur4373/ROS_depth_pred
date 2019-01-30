
from predict import *
# from depth import *
import numpy as np

print(out_norm_img)

for i in range(0,160):
    for j in range(0,128):

        delta = np.empty([128, 160, 3])
        delta(i,j)= max((depth(i,j)/depth_gt(i,j)),(depth_gt(i,j)/depth(i,j)))

    end
end


