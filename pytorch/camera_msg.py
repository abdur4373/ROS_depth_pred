from sensor_msgs.msg import CameraInfo, Image


def make_camera_msg(cam, rgb_time_sec, rgb_time_nsec, test_image):
    camera_info_msg = CameraInfo()
    camera_info_msg.header.seq = test_image

    if rgb_time_nsec + 20000000 > 999999999:
        camera_info_msg.header.stamp.secs = rgb_time_sec + 1
        camera_info_msg.header.stamp.nsecs = rgb_time_nsec + 20000000 - 100000000
    else:
        camera_info_msg.header.stamp.secs = rgb_time_sec
        camera_info_msg.header.stamp.nsecs = rgb_time_nsec + 20000000
    # camera_info_msg.header.stamp.secs = rgb_time_sec
    # camera_info_msg.header.stamp.nsecs = rgb_time_nsec + 20000000

    camera_info_msg.header.frame_id = "/openni_rgb_optical_frame"
    camera_info_msg.distortion_model = "plumb_bob"
    width, height = cam[0], cam[1]
    fx, fy = cam[2], cam[3]
    cx, cy = cam[4], cam[5]
    camera_info_msg.width = width
    camera_info_msg.height = height
    camera_info_msg.K = [fx, 0, cx,
                         0, fy, cy,
                         0, 0, 1]

    camera_info_msg.D = [0, 0, 0, 0, 0]

    camera_info_msg.R = [1, 0, 0,
                         0, 1, 0,
                         0, 0, 1]

    camera_info_msg.P = [fx, 0, cx, 0,
                         0, fy, cy, 0,
                         0, 0, 1, 0]

    return camera_info_msg


def make_rgb_msg(rgb, rgb_time_sec, rgb_time_nsec, test_image):
    rgb_msg = Image()

    rgb_msg.header.seq = test_image
    rgb_msg.header.frame_id = "/openni_rgb_optical_frame"
    rgb_msg.header.stamp.secs = rgb_time_sec
    rgb_msg.header.stamp.nsecs = rgb_time_nsec

    rgb_msg.step = 1920
    rgb_msg.is_bigendian = 0
    rgb_msg.encoding = 'rgb8'
    rgb_msg.data = rgb

    return rgb_msg


def make_depth_msg(depth, rgb_time_sec, rgb_time_nsec, test_image):
    depth_msg = Image()

    depth_msg.header.seq = test_image
    depth_msg.header.frame_id = "/openni_rgb_optical_frame"
    depth_msg.header.stamp.secs = rgb_time_sec
    depth_msg.header.stamp.nsecs = rgb_time_nsec - 20000000

    depth_msg.step = 2560
    depth_msg.is_bigendian = 0
    depth_msg.encoding = '32FC1'

    depth_msg.data = []

    return depth_msg
