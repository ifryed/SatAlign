import os

import cv2
import matplotlib.pyplot as plt


def main():
    path_dir = 'out/GoodRun/'
    log_path = os.path.join(path_dir, 'log.txt')
    base_img_path = os.path.join(path_dir, 'base.bmp')

    # plt.ion()
    img = plt.imread(os.path.join(path_dir, "{}.bmp".format(10)))
    with open(log_path, 'r') as log:
        for line in log:
            x, y, frame, has_frame = [int(x) for x in line.split(',')]
            if has_frame:
                img = plt.imread(os.path.join(path_dir, "{}.bmp".format(frame)))
            plt.cla()
            # plt.title(frame)
            plt.imshow(img**.5)
            plt.axis('off')
            circle1 = plt.Circle((x, y), 10, color='r', fill=False)
            plt.gca().add_patch(circle1)

            if has_frame:
                plt.savefig('out/figs/{}.png'.format(frame))
            # plt.pause(.01)

    # plt.show()


if __name__ == '__main__':
    os.makedirs('out/figs',exist_ok=True)
    [os.remove('out/figs/'+x) for x in os.listdir('out/figs/') if x.endswith('png')]
    main()
    print("DONE")
