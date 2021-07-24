import time
import cv2
import sys
import numpy as np
import picamera
import os
from BLD.Detector import BLDetector
from BLD.utils import Logger
import argparse

import threading

interval = 1 / 0.2

global kDST_FRQ
kDST_FRQ = 1.0

SENSOR_MODE = 7  # 640x480, partial FoV, binning 2x2
RESOLUTION = (640, 480)
FRAME_RATE = 120
fwidth = (RESOLUTION[0] + 31) // 32 * 32
fheight = (RESOLUTION[1] + 15) // 16 * 16


class RecordingOutput(object):
    """
    Object mimicking file-like object so start_recording will write each frame to it.
    See: https://picamera.readthedocs.io/en/release-1.12/api_camera.html#picamera.PiCamera.start_recording
    """

    def __init__(self, bld):
        self.bld = bld
        self.frame_c = 0
        self.st = time.time()
        self.logger = Logger('out/log.txt')
        self.wait_time = 1 / (2 * self.bld.freq)

    def write(self, buf):
        y_data = np.frombuffer(buf, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
        self.bld.addFrame(y_data)

        _, _, diff = self.bld.locateBlink()
        x, y = self.bld.getPoint()
        # avx, avy = self.bld.getAvgPoint()
        conf = self.bld.confidence()

        # Busy wait to ensure the right sampling of frames
        while time.time() - self.st < self.wait_time:
            pass

        et = time.time()
        c_fps = 1 / (et - self.st)
        self.st = et
        self.bld.updateFPS(c_fps)

        print("\r{}:\tFPS: {:.3f}\tConf: {:.3f}".format(self.frame_c, c_fps, conf), end='')
        if self.bld.buffFull():
            print("\t({:.3f},{:.3f})".format(x, y), end='')

        x, y = x * self.bld.scale, y * self.bld.scale  # Addapting the scale of the detection to the full frame size
        if self.frame_c % 10 == 0:
            cv2.imwrite('out/{}.bmp'.format(self.frame_c), y_data)
        self.logger.write(x, y, self.frame_c, self.frame_c % 10 == 0, conf)

        self.frame_c += 1

    def flush(self):
        pass  # called at end of recording


def main():
    h, w = 480, 640
    #    print("Frame Size: {}X{}".format(h,w))

    scale_factor = 3
    bld = BLDetector(kDST_FRQ, h, w, scale_factor, 2 * kDST_FRQ)
    print("Frame Size: {}x{}".format(bld.h_mini, bld.w_mini))

    with picamera.PiCamera(
            sensor_mode=SENSOR_MODE,
            resolution=RESOLUTION,
            framerate=FRAME_RATE) as camera:
        print('camera setup')
        time.sleep(2)
        print('starting recording')
        output = RecordingOutput(bld)

        camera.start_recording(output, 'yuv')
        # This while loop is here to let the detection loop run
        while bld.isRunning():
            pass

        camera.stop_recording()
        print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SatAlign params')
    parser.add_argument("-f", type=float, default=2.0,
                        help='The lasers blinking frequency')
    args = parser.parse_args()

    os.chdir('out')
    for f in os.listdir():
        os.remove(f)
    os.chdir('../')
    kDST_FRQ = args.f
    main()
