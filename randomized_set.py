
class RandomizedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        import random
        self.l = []
        self.d = dict()
        self.random = random.randint

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.d:
            self.l.append(val)
            self.d[val] = len(self.l) - 1
            return True
        else:
            return False

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if val not in self.d:
            return False
        i = self.d.pop(val)
        if i == len(self.l) - 1:
            self.l.pop()
        elif len(self.l):
            other_val = self.l.pop()
            self.d[other_val] = i
            self.l[i] = other_val
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return self.l[self.random(0, len(self.l) - 1)]

class RandomizedDuplicatedSet(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        import random
        self.l = []
        self.d = dict()
        self.random = random.randint

    def insert(self, val):
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        :type val: int
        :rtype: bool
        """
        self.l.append(val)
        if not self.d.get(val):
            self.d[val] = {len(self.l) - 1, }
            return True
        else:
            self.d[val].add(len(self.l) - 1)
            return False

    def remove(self, val):
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        :type val: int
        :rtype: bool
        """
        if not self.d.get(val):
            return False
        i = self.d[val].pop()
        if i == len(self.l) - 1:
            self.l.pop()
        elif len(self.l):
            other_val = self.l.pop()
            self.d[other_val].remove(len(self.l))
            self.d[other_val].add(i)
            self.l[i] = other_val
        return True

    def getRandom(self):
        """
        Get a random element from the set.
        :rtype: int
        """
        return self.l[self.random(0, len(self.l) - 1)]


rs = RandomizedDuplicatedSet()

rs.insert(1)
rs.insert(1)
rs.insert(1)
rs.insert(2)
rs.insert(3)
print rs.d, rs.l
rs.remove(1)
print rs.d, rs.l
rs.remove(1)
print rs.d, rs.l
rs.remove(1)
print rs.d, rs.l
rs.remove(3)
print rs.d, rs.l



