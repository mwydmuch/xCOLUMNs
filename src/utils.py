import time


class Timer(object):        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        print (f"  Time: {time.time() - self.start:>5.2f}s")
