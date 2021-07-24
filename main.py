import time
import cv2
import sys
import numpy as np
import picamera
import os
from BLD.Detector import BLDetector
from BLD.utils import Logger

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
    def __init__(self,bld):
        self.bld = bld
        self.frame_c = 0
        self.st = time.time()
        self.logger = Logger('out/log.txt')
        
    def write(self, buf):
        global frame_cnt, t_prev
        conf= 10000
        y_data = np.frombuffer(buf, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
        self.bld.addFrame(y_data)
        #y_data = np.frombuffer(buf, dtype=np.uint8, count=fwidth * fheight * 3).reshape((fheight, fwidth,3))
        
        _,_,diff = self.bld.locateBlink()
        x, y = self.bld.getPoint()
#             avx, avy = self.bld.getAvgPoint()
        conf = self.bld.confidence()

        while time.time()-self.st < 1/(2*self.bld.freq):
           pass
        
        et = time.time()
        c_fps = 1 / (et - self.st)
        self.st = et
        self.bld.updateFPS(c_fps)
        
        print("\r{}:\tFPS: {:.3f}\tConf: {:.3f}".format(self.frame_c, c_fps, conf), end='')
        if False and self.bld.buffFull():
            print("\t({:.3f},{:.3f})".format(x, y), end='')

        if True or conf < 5:
            print()
#         if self.bld.buffFull() and (self.frame_c % 10 == 0):
# #             f=cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 5)
# #             f = f.get()
#             cv2.imwrite('out/{}.bmp'.format(self.frame_c), 255*frame1)
#            f=y_data.copy()
            x,y = x*self.bld.scale,y*self.bld.scale
#            f[int(y-10):int(y+10),int(x-10):int(x+10)]=255
#            cv2.imwrite('out/{}.bmp'.format(self.frame_c), (255*diff/diff.max()).astype(np.uint8))
            if self.frame_c % 10 == 0:
                cv2.imwrite('out/{}.bmp'.format(self.frame_c),y_data)
#            print("{:.0f},{:.0f}:{:.3f}".format(x,y,conf))
            self.logger.write(x,y,self.frame_c,self.frame_c%10 == 0)
        
        self.frame_c += 1
        
    def flush(self):
        pass  # called at end of recording


def main(video_path: any = 0):
    global st, c

    h,w = 480, 640
#    print("Frame Size: {}X{}".format(h,w))
    
    scale_factor = 3
    bld = BLDetector(kDST_FRQ, h, w, scale_factor, 30.)
    print("Frame Size: {}x{}".format(bld.h_mini,bld.w_mini))

    st = time.time()
    c = 0

    with picamera.PiCamera(
        sensor_mode=SENSOR_MODE,
        resolution=RESOLUTION,
        framerate=FRAME_RATE) as camera:
        
        print('camera setup')
        time.sleep(2)  
        print('starting recording')
        output = RecordingOutput(bld)
        
        camera.start_recording(output, 'yuv')
#         camera.wait_recording(3)
        while True:
            pass
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#                 
            

#             if bld.buffFull() and (c % 10 == 0):
#                 f=cv2.circle(frame, (int(x), int(y)), 20, (0, 255, 0), 5)
#                 f = f.get()
#                 cv2.imwrite('out/{}.png'.format(time.time()), f[:, :, [2, 1, 0]])

        
        
        camera.stop_recording()
        print("Done")
        
        


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
    os.chdir('out')
    for f in os.listdir():
        os.remove(f)
    os.chdir('../')
    kDST_FRQ = 2
    main()
