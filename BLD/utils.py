class Logger():
    kFLUSH_CONST = 10

    def __init__(self, path):
        self.path = path
        self.fd = open(self.path, 'w')
        self.flush_counter = Logger.kFLUSH_CONST

    def write(self, x, y, time,has_image):
        entry_str = "{:.0f},{:.0f},{:.0f},{}\n".format(x, y, time,int(has_image))
        self.fd.write(entry_str)

        self.flush_counter -= 1
        if self.flush_counter == 0:
            self.fd.flush()
            self.flush_counter = Logger.kFLUSH_CONST

    def close(self):
        self.fd.flush()
        self.fd.close()

    def __del__(self):
        self.close()
