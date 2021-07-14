import time

import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import skimage.measure

from BLD.Detector import BLDetector

import threading

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
    plt.ion()
    video = cv2.VideoCapture(video_path)

    fps = video.get(cv2.CAP_PROP_FPS)
    print("Video FPS: {:.3f}".format(fps))
    video.set(3, 640)
    video.set(4, 480)

    h, w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))//4, int(video.get(cv2.CAP_PROP_FRAME_WIDTH))//4
    print(h, w)
    scale_factor = 1
    bld = BLDetector(kDST_FRQ, h, w, scale_factor, fps)

    st = time.time()
    c = 0

    while video.isOpened():
        loop_time = time.time()
        # def mainLoop():
        #     global st,c
        #     threading.Timer(interval, mainLoop).start()
        ret, frame = video.read()
        frame = cv2.resize(frame,(0,0),fx=.25,fy=0.25)

        if not ret:
            video.release()
            break

        frame = frame[:, :, [2, 1, 0]]

        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
            break

        bld.addFrame(frame)

        if False or bld.buffFull():
            plt.cla()
            _, _, frame1 = bld.locateBlink()
            # print(bld.getAzimut())
            x, y = bld.getPoint()
            avx, avy = bld.getAvgPoint()

            disp_frame = frame
            plt.imshow(disp_frame, cmap='gray')
            plt.plot(y * scale_factor, x * scale_factor, '*r')
            plt.plot(avy * scale_factor, avx * scale_factor, 'xb')
            conf = bld.confidence()
            conf = "Confidence: {:.3f}".format(conf)
            plt.title(conf, color='red', fontsize=20, weight='bold')
            # plt.text(100, 100, conf, color='red', fontsize=20, weight='bold')
            plt.pause(0.1)

        et = time.time()
        c_fps = 1 / (et - st)
        print("\r{}: FPS: {:.3f} Conf: {:.3f}".format(c, c_fps, bld.confidence()), end='')
        if bld.buffFull():
            print("\t({:.3f},{:.3f})".format(x, y), end='')
        st = time.time()
        c += 1

        # if bld.buffFull() and (c % 10 == 0):
        #     f=cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 5)
        #     f = f.get()
        #     cv2.imwrite('out/{}.png'.format(time.time()), f[:, :, [2, 1, 0]])

        bld.updateFPS(c_fps)
        while time.time() - loop_time < interval:
            pass
    # mainLoop()

    # video.release()


if __name__ == '__main__':
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

    kDST_FRQ = 0.1
    main(1)
