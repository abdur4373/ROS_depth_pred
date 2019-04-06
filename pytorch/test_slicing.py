from cv_bridge import CvBridge
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


# import cv_bridge

# print(cv_bridge.__version__)
#
# path_to_depth = '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/test.mat'
# # image_path= '/home/maq/PycharmProjects/Deeper-Depth-Prediction_2/pytorch/test_image.jpg'
# #read mat file
# f = h5py.File(path_to_depth)
#
# # read 0-th image. original format is [3 x 640 x 480], uint8
# img = f['rgb_undist'][1]
# # # reshape
# img_ = np.empty([480, 640, 3])
# img_[:, :, 0] = img[0, :, :].T
# img_[:, :, 1] = img[1, :, :].T
# img_[:, :, 2] = img[2, :, :].T
# img__ = img_.astype('uint8')
#
# print("test image from mat file", img__.dtype)

# io.imsave("test_image.jpg", img__ / 255.0)

# image = cv2.imread("test_image.jpg")
# print("test image from jpg file", image.dtype)
#
# cv2.imshow('cv_image', image)
# msg = CvBridge().cv2_to_imgmsg(img__, encoding="bgr8")


def callback_rgb(data):
    cv_image = CvBridge().imgmsg_to_cv2(data)
    # io.imsave("test_image_2.jpg", cv_image / 255.0)
    rospy.loginfo("I heard rgb image %s", cv_image.dtype)
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)


def callback_depth(data):
    cv_image = CvBridge().imgmsg_to_cv2(data)
    # depth = np.array(cv_image)
    # depth_good_entries=np.count_nonzero(~np.isnan(depth))
    # depth_good_percent= depth_good_entries*100/depth.size
    rospy.loginfo("I heard depth data %s", cv_image.dtype)
    # cv2.imshow("Image window", cv_image)
    # cv2.waitKey(3)


if __name__ == '__main__':
    sub_rgb = rospy.Subscriber('/camera/rgb/image_color', Image, callback=callback_rgb)
    sub_depth = rospy.Subscriber('/camera/depth/image', Image, callback=callback_depth)

    rospy.init_node('image', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

# while not rospy.is_shutdown():
#     rate.sleep()


# pub=rospy.Publisher('rgb', Image)
# rospy.init_node('image', anonymous=True)
# rate=rospy.Rate(0.5)
#
# while not rospy.is_shutdown():
#     pub.publish(msg)
#     rate.sleep()
