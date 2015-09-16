from output import OutputStream
from threading import Thread
from time import sleep

class BackgroundWorker():
    """Background Worker class. Runs a thread that does work. Needs to be subclassed.
Call this classes' __init__ before anything else in the subclass __init__.
Override dowork(self) to do actual work. This should be broken up into small pieces to allow
the thread to check whether it should quit and whether it is paused.
"""
    def __init__(self, os=None):
        self.running = False    # true if worker should be stopped (i.e. killing optimizer)
        self.paused = False     # true if worker should be paused

        if os==None:
            self.os = OutputStream()    # default to stdout
        else:
            self.os = os            # OutputStream where to write files

    def dowork(self):
        self.os("  Thread doing stuff..")
        sleep(1)

    def worker(self):
        self.os("Thread Started!")
        r = True
        while (r == None or r == True) and self.running:
            if not self.paused:
                r = self.dowork()
            else:
                pass

    def begin(self):
        if not self.running:
            self.running = True
            self.paused = False

            self.thread = Thread(target=self.worker)
            self.thread.start()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def end(self):
        self.pause()
        self.os("**** Joining Thread ****")
        self.running = False
        self.thread.join()


