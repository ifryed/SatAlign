import numpy as np
from cv2 import cvtColor, resize, filter2D, COLOR_BGR2HSV, COLOR_RGB2HSV


class BLDetector():
    def __init__(self, freq, h, w, buff_scale=3, fps=30):
        self.freq = freq
        self.fps = fps
        self.buffer_len = 5
        self.scale = buff_scale
        self.h_mini, self.w_mini = h // buff_scale, w // buff_scale
        self.buffer = np.zeros((self.h_mini, self.w_mini, self.buffer_len))

        ## init
        self.counter = 0
        self.first_time = True

        self.x_lst, self.y_lst = np.array([0]), np.array([0])
        self.mom_x, self.mom_y = 0, 0
        self.x_mean, self.y_mean = 100, 100
        self.avg_x, self.avg_y = 0, 0

        self.stopped = False

    def updateFPS(self, n_fps):
        """
        Updates the FPS based on RT readings
        @param n_fps: The last FPS
        @return: None
        """
        alpha = 1 / self.buffer_len
        self.fps = self.fps * alpha + (1 - alpha) * n_fps

    def addFrame(self, frame):
        """
        Added a new image to the image buffer
        @param frame:
        @return: None
        """
        # q_factor = 1
        # frame = (frame // q_factor) * q_factor
        curr_idx = self.counter % self.buffer_len
        frame_mini = resize(frame, (self.w_mini, self.h_mini))

        self.buffer[:, :, curr_idx] = frame_mini
        self.counter += 1

    def _updateLocation(self):
        """
        Updates the current belived location of the laser
        @return: None
        """
        alpha = 0.8
        m_alpha = 0.5
        if len(self.x_lst) == 0:
            self.x_mean, self.y_mean = 0, 0
        else:
            self.x_mean, self.y_mean = np.mean(self.x_lst), np.mean(self.y_lst)

        # Momentum
        self.mom_x = m_alpha * self.mom_x - alpha * (self.avg_x - self.x_mean)
        self.mom_y = m_alpha * self.mom_y - alpha * (self.avg_y - self.y_mean)

        self.avg_x += self.mom_x
        self.avg_y += self.mom_y

    def getFrame(self):
        """
        Returns the last frame
        @return: Last frame
        """
        curr_idx = (self.counter - 1) % self.buffer_len
        return self.buffer[:, :, curr_idx]

    def locateBlink(self):
        """
        Calculates the location of the laser in the image.
        @return: possible locations in X, Y, differance frame (for debug)
        """
        if not self.buffFull():
            return None, None, None
        # b_offset = max(1,int(self.fps/self.freq/2))
        b_offset = 1

        diff = np.abs(self.buffer[:, :, :-b_offset] - self.buffer[:, :, b_offset:])
        diff_mean = diff.mean(axis=2)

        freq_mat = diff_mean > diff_mean.max() * 0.95
        self.y_lst, self.x_lst = np.where(freq_mat)

        self._updateLocation()
        return self.x_lst, self.y_lst, diff_mean[:]

    def getPoint(self):
        """
        Returns current location
        @return: Coordinates-> x,y
        """
        return [self.x_mean, self.y_mean]

    def getAvgPoint(self):
        """
        Returns \textit{ghost} location.
        @return: Coordinates-> x,y
        """
        return [self.avg_x, self.avg_y]

    def confidence(self):
        """
        Calculates the confidence of the current location.
        The confidence is based on variance and momentum.
        @return: Location confidence
        """
        if len(self.y_lst) == 0:
            return 10000000
        var = self.y_lst.var() + self.x_lst.var()
        diff = np.abs(self.mom_x) + np.abs(self.mom_y)

        conf = var + diff

        if conf < 1:
            self.stopped = True
        return conf

    def getAzimut(self):
        """
        Returns the location of the detection in radians
        @return: \theta _x,\theta _y
        """
        y, x = self.avg_y, self.avg_x
        return np.arctan2(self.w_mini, (x - self.h_mini / 2)), \
               np.arctan2(self.h_mini, (y - self.w_mini / 2))

    def buffFull(self):
        """
        Returns the status of the buffer
        @return: Returns the status of the buffer
        """
        return self.counter > self.buffer_len

    def isRunning(self):
        """
        Returns if the detection is still running
        @return: Returns if the detection is still running
        """
        return self.stopped
