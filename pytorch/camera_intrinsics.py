import numpy as np
from sensor_msgs.msg import CameraInfo, RegionOfInterest
from std_msgs.msg import Header


class CameraIntrinsics(object):
    """A set of intrinsic parameters for a camera. This class is used to project
    and deproject points.
    """

    def __init__(self, frame, fx, fy=None, cx=0.0, cy=0.0, skew=0.0, height=None, width=None):
        """Initialize a CameraIntrinsics model.
        Parameters
        ----------
        frame : :obj:`str`
            The frame of reference for the point cloud.
        fx : float
            The x-axis focal length of the camera in pixels.
        fy : float
            The y-axis focal length of the camera in pixels.
        cx : float
            The x-axis optical center of the camera in pixels.
        cy : float
            The y-axis optical center of the camera in pixels.
        skew : float
            The skew of the camera in pixels.
        height : float
            The height of the camera image in pixels.
        width : float
            The width of the camera image in pixels
        """
        self._frame = frame
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)
        self._skew = float(skew)
        self._height = int(height)
        self._width = int(width)

        # set focal, camera center automatically if under specified
        if fy is None:
            self._fy = fx

        # set camera projection matrix
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def frame(self):
        """:obj:`str` : The frame of reference for the point cloud.
        """
        return self._frame

    @property
    def fx(self):
        """float : The x-axis focal length of the camera in pixels.
        """
        return self._fx

    @property
    def fy(self):
        """float : The y-axis focal length of the camera in pixels.
        """
        return self._fy

    @property
    def cx(self):
        """float : The x-axis optical center of the camera in pixels.
        """
        return self._cx

    @cx.setter
    def cx(self, z):
        self._cx = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def cy(self):
        """float : The y-axis optical center of the camera in pixels.
        """
        return self._cy

    @cy.setter
    def cy(self, z):
        self._cy = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def skew(self):
        """float : The skew of the camera in pixels.
        """
        return self._skew

    @property
    def height(self):
        """float : The height of the camera image in pixels.
        """
        return self._height

    @property
    def width(self):
        """float : The width of the camera image in pixels
        """
        return self._width

    @property
    def proj_matrix(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @property
    def K(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @property
    def vec(self):
        """:obj:`numpy.ndarray` : Vector representation for this camera.
        """
        return np.r_[self.fx, self.fy, self.cx, self.cy, self.skew, self.height, self.width]

    @property
    def rosmsg(self):
        """:obj:`sensor_msgs.CamerInfo` : Returns ROS CamerInfo msg
        """
        msg_header = Header()
        msg_header.frame_id = self._frame

        msg_roi = RegionOfInterest()
        msg_roi.x_offset = 0
        msg_roi.y_offset = 0
        msg_roi.height = 0
        msg_roi.width = 0
        msg_roi.do_rectify = 0

        msg = CameraInfo()
        msg.header = msg_header
        msg.height = self._height
        msg.width = self._width
        msg.distortion_model = 'plumb_bob'
        msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
        msg.K = [self._fx, 0.0, self._cx, 0.0, self._fy, self._cy, 0.0, 0.0, 1.0]
        msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.P = [self._fx, 0.0, self._cx, 0.0, 0.0, self._fx, self._cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        msg.binning_x = 0
        msg.binning_y = 0
        msg.roi = msg_roi
        print msg
        return msg


CameraIntrinsics()
