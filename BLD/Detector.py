import numpy as np
from cv2 import cvtColor, resize, filter2D, COLOR_BGR2HSV, COLOR_RGB2HSV

import matplotlib.pyplot as plt

COLOR_BGR2GRAY = 6


class BLDetector():
    def __init__(self, freq, h, w, buff_scale=3, fps=30):
        self.freq = freq
        self.fps = fps
        self.buffer_len = 10
        self.h_mini, self.w_mini = h // buff_scale, w // buff_scale
        self.buffer = np.zeros((self.h_mini, self.w_mini, self.buffer_len))
        self.img_buffer = np.zeros((h, w, self.buffer_len))

        ## init
        self.counter = 0
        self.first_time = True

        self.x_lst, self.y_lst = np.array([0]), np.array([0])
        self.mom_x, self.mom_y = 0, 0
        self.x_mean, self.y_mean = 100, 100
        self.avg_x, self.avg_y = 0, 0

    def updateFPS(self, n_fps):
        alpha = 0.6
        self.fps = self.fps * alpha + (1 - alpha) * n_fps

    def addFrame(self, frame):
        q_factor = 20
        # gray = cvtColor(frame, COLOR_BGR2GRAY).astype(np.float32)
        hsv = cvtColor(frame, COLOR_RGB2HSV).astype(np.float32)
        hue, _, gray = [x.squeeze() for x in np.split(hsv, 3, axis=2)]
        # gray = 255*gray/gray.max
        mask = (hue < 20) + (hue > 160)
        mask = mask * (gray > 40)
        # gray = gray * mask
        gray = (gray // q_factor) * q_factor
        curr_idx = self.counter % self.buffer_len
        if self.first_time:
            self.first_time = False
            self.img_buffer[:, :, curr_idx] = gray
            self.img_buffer[:, :, -1] = gray

        img_idx = self.counter - 1
        first_img = self.img_buffer[:, :, img_idx % self.buffer_len]
        # aligned_fs = align(gray, first_img)
        aligned_fs = gray
        aligned = resize(aligned_fs, (self.w_mini, self.h_mini))

        self.img_buffer[:, :, curr_idx] = aligned_fs
        self.buffer[:, :, curr_idx] = np.square(aligned)
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
        b_mean = self.buffer.mean(axis=2)
        buff_diff = self.buffer - b_mean[..., np.newaxis]
        buff_dig = (buff_diff > 0) - 0.5

        freq_mat = self._getFreq(buff_dig)
        f_2d = (np.abs(freq_mat - self.freq) <= .03).astype(np.float32)
        k_size = 7
        filt = filter2D(f_2d, -1, np.ones((k_size, k_size)))

        filt_max = sorted(filt[::2, ::2].flatten())
        self.x_lst, self.y_lst = np.where(filt >= max(1, filt_max[-4]))

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
        # diff = np.sqrt(np.square(self.x_mean - self.avg_x) + np.square(self.y_mean - self.avg_y))
        diff = np.abs(self.mom_x) + np.abs(self.mom_y)
        # print(np.abs(self.mom_x) + np.abs(self.mom_y))

        conf = var + diff
        return conf

    def getAzimut(self):
        y, x = self.avg_y, self.avg_x
        return np.arctan2(self.h_mini, (y - self.h_mini / 2)), \
               np.arctan2(self.w_mini, (x - self.w_mini / 2))

    def buffFull(self):
        return self.counter > self.buffer_len

    def _getFreq(self, signal):
        """
        https://stackoverflow.com/questions/55283610/how-do-i-get-the-frequencies-from-a-signal
        """

        n = signal.shape[-1]
        k = np.arange(n)
        T = n / self.fps
        frq = k / T
        frq = frq[: len(frq) // 2]

        Y = np.fft.fft(signal) / n
        Y = Y[:, :, :n // 2]

        max_arg = abs(Y).argmax(axis=2)
        freq_2d = frq[max_arg]
        return freq_2d
