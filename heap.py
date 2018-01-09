class Heap(object):
    def __init__(self, compare_func=lambda a, b: a < b):
        self._cmp = compare_func
        self._list = []

    def is_empty(self):
        return len(self._list) == 0

    def __str__(self):
        return str(self._list)

    @property
    def top(self):
        if len(self._list):
            return self._list[0]
        else:
            return None

    def pop_top(self):
        if len(self._list):
            top = self._list[0]
            self._list[0] = self._list[-1]
            self._list.pop()
            curr = 0
            while True:
                smaller_son = None
                if 2 * curr + 1 < len(self._list):
                    if self._cmp(self._list[2 * curr + 1], self._list[curr]):
                        smaller_son = 2 * curr + 1
                    if 2 * curr + 2 < len(self._list):
                        if smaller_son:
                            if self._cmp(self._list[2 * curr + 2], self._list[smaller_son]):
                                smaller_son = 2 * curr + 2
                        else:
                            if self._cmp(self._list[2 * curr + 2], self._list[curr]):
                                smaller_son = 2 * curr + 2
                if smaller_son is not None:
                    self._list[curr], self._list[smaller_son] = self._list[smaller_son], self._list[curr]
                    curr = smaller_son
                else:
                    break
            return top

    def insert(self, val):
        self._list.append(val)
        curr = len(self._list) - 1
        while True:
            parent = None
            if (curr - 1) / 2 >= 0:
                if not self._cmp(self._list[(curr - 1) / 2], self._list[curr]):
                    parent = (curr - 1) / 2
            if parent is not None:
                self._list[curr], self._list[parent] = self._list[parent], self._list[curr]
                curr = parent
            else:
                break
        self._list[curr] = val


if __name__ == '__main__':
    h = Heap()
    h.insert(-5)
    h.insert(5)
    h.insert(4)
    h.insert(-4)
    h.insert(4)
    h.insert(4)

    print h.pop_top()
    print h.pop_top()
    print h.pop_top()
    print h.pop_top()


    h.insert(4)
    h.insert(-3)
    h.insert(3)
    h.insert(-3)
    h.insert(2)
    h.insert(2)
    h.insert(-1)

    print h.pop_top()