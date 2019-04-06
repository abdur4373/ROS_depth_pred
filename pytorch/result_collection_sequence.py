# import skimage.io as io
import rospy
from cv_bridge import CvBridge

from camera_msg import *
from predict import *

if __name__ == '__main__':
    rospy.init_node('depth_prediction_node')
    while ~rospy.is_shutdown():
        for test_image in range(200):
            # if ~rospy.is_shutdown():
            print(test_image)

            t = time.time()

            rgb_image, depth_gt, rgb_time_sec, rgb_time_nsec = main_mod(int(test_image))

            cam = [640, 480, 518.86, 519.47, 325.58, 253.74]
            camera_info_msg = make_camera_msg(cam, rgb_time_sec, rgb_time_nsec, test_image)

            # print(camera_info_msg)

            # print rgb_image.dtype, depth_gt.dtype

            # print("depth image from mat file", depth_gt.dtype)

            # depth = CvBridge().cv2_to_imgmsg(depth_gt, encoding="32FC1")

            # depth_msg = make_depth_msg(depth_gt, rgb_time_sec, rgb_time_nsec, test_image)

            depth_msg = CvBridge().cv2_to_imgmsg(depth_gt, encoding="32FC1")
            depth_msg.header.frame_id = "/openni_rgb_optical_frame"
            depth_msg.header.stamp.secs = rgb_time_sec
            depth_msg.header.stamp.nsecs = rgb_time_nsec
            depth_msg.header.seq = test_image

            pub_depth = rospy.Publisher('/camera/depth/image', Image, queue_size=1)
            pub_depth.publish(depth_msg)
            print("depth msg is:", depth_msg.encoding, depth_msg.is_bigendian, depth_msg.step, depth_msg.header)

            rgb_image = rgb_image.astype('uint8')

            print("rgb image from mat file", rgb_image.dtype)
            rgb_msg = CvBridge().cv2_to_imgmsg(rgb_image, encoding="rgb8")
            rgb_msg.header.frame_id = "/openni_rgb_optical_frame"
            if rgb_time_nsec + 20000000 > 999999999:
                rgb_msg.header.stamp.secs = rgb_time_sec + 1
                rgb_msg.header.stamp.nsecs = rgb_time_nsec + 20000000 - 100000000
            else:
                rgb_msg.header.stamp.secs = rgb_time_sec
                rgb_msg.header.stamp.nsecs = rgb_time_nsec + 20000000

            rgb_msg.header.seq = test_image
            # rgb_msg = make_rgb_msg(rgb, rgb_time_sec, rgb_time_nsec, test_image)

            print("rgb msg is:", rgb_msg.encoding, rgb_msg.is_bigendian, rgb_msg.step, rgb_msg.header)

            pub_camera_info = rospy.Publisher('/camera/rgb/camera_info', CameraInfo, queue_size=1)
            pub_rgb = rospy.Publisher('/camera/rgb/image_color', Image, queue_size=1)
            # rospy.init_node('image', anonymous=True)
            # rate = rospy.Rate(0.5)

            pub_camera_info.publish(camera_info_msg)
            pub_rgb.publish(rgb_msg)

            print("Finished publishing image in {0} s".format(time.time() - t))

            # time.sleep(2)
        # else:
        #     rospy.signal_shutdown()

    rospy.spin()
