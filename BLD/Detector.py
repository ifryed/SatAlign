import numpy as np
from cv2 import cvtColor, resize, filter2D, COLOR_BGR2HSV, COLOR_RGB2HSV, COLOR_RGB2GRAY

import matplotlib.pyplot as plt

COLOR_BGR2GRAY = 6

NMS_KERNEL = -np.ones((7, 7))
NMS_KERNEL[2:-2, 2:-2] = 1


class BLDetector():
    def __init__(self, freq, h, w, buff_scale=3, fps=30):
        self.freq = freq
        self.fps = fps
        self.buffer_len = 5
        self.h_mini, self.w_mini = h // buff_scale, w // buff_scale
        self.scale_x, self.scale_y = h / self.h_mini, w / self.w_mini
        self.buffer = np.zeros((self.h_mini, self.w_mini, self.buffer_len))
        self.img_buffer = np.zeros((h, w, self.buffer_len))

        ## init
        self.counter = 0
        self.first_time = True

        self.x_lst, self.y_lst = np.array([0]), np.array([0])
        self.mom_x, self.mom_y = 0, 0
        self.x_mean, self.y_mean = 100, 100
        self.avg_x, self.avg_y = 0, 0
        self.sensitivity = self.freq * 0.2

    def updateFPS(self, n_fps):
        alpha = 1 / self.buffer_len
        self.fps = self.fps * alpha + (1 - alpha) * n_fps

    def addFrame(self, frame):
        gray = cvtColor(frame, COLOR_RGB2GRAY).astype(np.float32)
        curr_idx = self.counter % self.buffer_len
        mini_gray = resize(gray, (self.w_mini, self.h_mini))

        self.buffer[:, :, curr_idx] = mini_gray
        self.counter += 1

    def _updateLocation(self):
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
        curr_idx = (self.counter - 1) % self.buffer_len
        return self.img_buffer[:, :, curr_idx]

    def locateBlink(self):
        if not self.buffFull():
            return None
        # b_offset = max(1, int(self.fps / self.freq / 2))
        b_offset = 1

        diff = np.abs(self.buffer[:, :, :-b_offset] - self.buffer[:, :, b_offset:])
        diff_mean = diff.mean(axis=2)

        # diff_mean = filter2D(diff_mean, -1, NMS_KERNEL)

        freq_mat = diff_mean > diff_mean.max() * .95
        self.x_lst, self.y_lst = np.where(freq_mat)

        self._updateLocation()
        return self.x_lst, self.y_lst, freq_mat

    def getPoint(self):
        return [self.x_mean, self.y_mean]

    def getAvgPoint(self):
        return [self.avg_x, self.avg_y]

    def confidence(self):
        if len(self.y_lst) == 0:
            return 10000000
        var = self.y_lst.var() + self.x_lst.var()
        diff = np.square(self.mom_x) + np.square(self.mom_y)

        conf = var + diff
        return conf

    def getAzimut(self):
        y, x = self.avg_y, self.avg_x
        return np.arctan2(self.h_mini, (y - self.h_mini / 2)), \
               np.arctan2(self.w_mini, (x - self.w_mini / 2))

    def buffFull(self):
        return self.counter > self.buffer_len
