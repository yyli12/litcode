class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        from collections import deque
        self.cap = capacity
        self.dic = dict()
        self.queue = deque()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        self.queue.append(key)
        return self.dic.get(key, -1)

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if len(self.dic) >= self.cap:
            lru = self.queue.popleft()
            while lru not in self.cap:
                lru = self.queue.popleft()
            self.cap.pop(lru)
        self.queue.append(key)
        self.dic[key] = value


obj = LRUCache(2)
obj.put(1,1)
obj.put(2,2)