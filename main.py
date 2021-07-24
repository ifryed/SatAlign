import time

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import skimage.measure

from BLD.Detector import BLDetector

import threading

from BLD.utils import Logger

interval = 1 / 0.2

global kDST_FRQ
kDST_FRQ = 1.0


def displayPano(img1, img2, p1, p2):
    h, w = img1.shape[:2]
    canvas = np.zeros((h, 2 * w))
    canvas[:, :w] = img1
    canvas[:, -w:] = img2

    plt.imshow(canvas, cmap='gray')
    plt.plot(p1[:, 0], p1[:, 1], '*r')
    plt.plot(p2[:, 0] + w, p2[:, 1], '*g')

    for pa, pb in zip(p1, p2):
        plt.plot([pa[0], w + pb[0]], [pa[1], pb[1]], '-b')

    plt.show()


def main(video_path: any = 0):
    global st, c
    # plt.ion()
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    print("Video FPS: {:.3f}".format(fps))
    video.set(3, 640)
    video.set(4, 480)

    h, w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    print(h, w)
    scale_factor = 1
    bld = BLDetector(kDST_FRQ, h, w, scale_factor, fps)

    st = time.time()
    c = 0
    conf = 1000

    logger = Logger('out/log.txt')
    ret, frame = video.read()
    cv2.imwrite('out/base.bmp', frame)
    while video.isOpened():
        st = time.time()
        ret, frame = video.read()

        frame = frame[:, :, [2, 1, 0]]

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        bld.addFrame(frame)
        bld.locateBlink()
        x, y = bld.getPoint()
        conf = bld.confidence()

        while time.time() - st < 1 / (2 * kDST_FRQ):
            pass
        et = time.time()
        c_fps = 1 / (et - st)
        bld.updateFPS(c_fps)
        # print("\r{}:\tFPS: {:.1f} Conf: {:.2f}\t{:.3f},{:.3f}".format(c, c_fps, conf, x, y), end='')
        c += 1
        if True or conf < 10:
            # print()
            # print("Conf:{:.2f}\t{:.3f},{:.3f}".format(conf, x, y))
            plt.cla()
            plt.imshow(frame)
            plt.plot(y, x, 'og')
            # plt.plot(avy, avx, 'xb')
            plt.pause(.01)
            logger.write(y, x, c, c % 10 == 0)
            if c % 10 == 0:
                cv2.imwrite('out/{}.bmp'.format(c), frame)

    logger.close()
    video.release()


if __name__ == '__main__':
    plt.ion()
    # kDST_FRQ = 1
    # main('data/1hz_cut.mp4')
    # kDST_FRQ = 5
    # main('data/5hz_cut.mp4')
    # kDST_FRQ = 10
    # main('data/10hz_cut.mp4')
    # kDST_FRQ = 2
    # main('data/laser_blink_cut.mp4')
    # kDST_FRQ = 2
    # main('data/laser_blink_bed.mp4')

    kDST_FRQ = 2
    main(0)
