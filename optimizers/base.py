class BaseStep():
    def __init__(self):
        pass

    def setup(self,cparams,diagout,statout,ostream):
        self.diagout = diagout
        self.statout = statout
        self.ostream = ostream

