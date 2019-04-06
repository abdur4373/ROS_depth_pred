from sensor_msgs.msg import CameraInfo


def make_camera_msg(cam):
    camera_info_msg = CameraInfo()
    # camera_info_msg.header.seq=
    # camera_info_msg.header.stamp.secs
    # camera_info_msg.header.stamp.nsecs
    # camera_info_msg.header.frame_id
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


cam = [640, 480, 518.86, 519.47, 325.58, 253.74]
camera_info_msg = make_camera_msg(cam)

print(camera_info_msg)
