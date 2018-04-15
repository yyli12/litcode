import bisect

class RangeModule(object):

    def __init__(self):
        self.X = [0, 10**9]
        self.track = [False] * 2

    def addRange(self, left, right, track=True):
        def index(x):
            i = bisect.bisect_left(self.X, x)
            if self.X[i] != x:
                self.X.insert(i, x)
                self.track.insert(i, self.track[i-1])
            return i
        i = index(left)
        j = index(right)
        print i, j
        self.X[i:j] = [left]
        self.track[i:j] = [track]

    def queryRange(self, left, right):
        i = bisect.bisect(self.X, left) - 1
        j = bisect.bisect_left(self.X, right)
        return all(self.track[i:j])

    def removeRange(self, left, right):
        self.addRange(left, right, False)


def test():
    rm = RangeModule()
    rm.addRange(1, 10)
    print rm.X, rm.track
    rm.addRange(110, 120)
    print rm.X, rm.track
    rm.removeRange(1,121)
    print rm.X, rm.track


if __name__ == '__main__':
    test()