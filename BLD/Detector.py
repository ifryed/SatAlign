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
        self.img_buffer = np.zeros((h, w, self.buffer_len))

        ## init
        self.counter = 0
        self.first_time = True

        self.x_lst, self.y_lst = np.array([0]), np.array([0])
        self.mom_x, self.mom_y = 0, 0
        self.x_mean, self.y_mean = 100, 100
        self.avg_x, self.avg_y = 0, 0

    def updateFPS(self, n_fps):
        alpha = 1/self.buffer_len
        self.fps = self.fps * alpha + (1 - alpha) * n_fps

    def addFrame(self, frame):
        #q_factor = 1
        #frame = (frame // q_factor) * q_factor
        curr_idx = self.counter % self.buffer_len
        frame_mini = resize(frame, (self.w_mini, self.h_mini))

        self.buffer[:, :, curr_idx] = frame_mini
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
    
    def getFrame2(self):
        curr_idx = (self.counter - 1) % self.buffer_len
        return self.buffer[:, :, curr_idx]

    def locateBlink(self):
        if not self.buffFull():
            return None,None,None
        #b_offset = max(1,int(self.fps/self.freq/2))
        b_offset = 1

        diff = np.abs(self.buffer[:,:,:-b_offset] - self.buffer[:,:,b_offset:])
        diff_mean = diff.mean(axis=2)

        freq_mat = diff_mean > diff_mean.max() * 0.95
        self.y_lst, self.x_lst = np.where(freq_mat)

        self._updateLocation()
        return self.x_lst, self.y_lst, diff_mean[:]

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
        return np.arctan2(self.w_mini, (x - self.h_mini / 2)), \
               np.arctan2(self.h_mini, (y - self.w_mini / 2))

    def buffFull(self):
        return self.counter > self.buffer_len

    def _getFreq(self, signal):
        """
        https://stackoverflow.com/questions/55283610/how-do-i-get-the-frequencies-from-a-signal
        """

        n = signal.shape[-1]
        T = n / self.fps
        frq = np.arange(n) / T
        frq = frq[: len(frq) // 2]

        Y = np.fft.fft(signal) / n
        Y = Y[:, :, :n // 2]

        max_arg = np.abs(Y).argmax(axis=2)
        freq_2d = frq[max_arg]
        return freq_2d
