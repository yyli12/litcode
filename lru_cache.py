class LRUCache(object):

    class DoublyLinkedListNode(object):

        def __init__(self, key, val):
            self.key = key
            self.val = val
            self.nxt = None
            self.prv = None

        def __str__(self):
            return '%s(%s)-%s' % (self.key, self.val, self.nxt)

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cap = capacity
        self.dic = dict()
        self.queue = self.DoublyLinkedListNode(0, 0)
        self.tail = self.queue

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dic:
            node = self.dic[key]
            self.remove(node)
            self.tail.nxt = node
            node.prv = self.tail
            self.tail = self.tail.nxt
            return node.val
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """
        if key in self.dic:
            node = self.dic[key]
            self.remove(node)
            node.val = value
        else:
            if len(self.dic) >= self.cap:
                to_delete = self.queue.nxt
                self.remove(to_delete)
                self.dic.pop(to_delete.key)
            node = self.DoublyLinkedListNode(key, value)
        self.tail.nxt = node
        node.prv = self.tail
        self.tail = self.tail.nxt
        self.dic[key] = node

    def remove(self, node):
        node.prv.nxt = node.nxt
        if node.nxt:
            node.nxt.prv = node.prv
        else:
            self.tail = self.tail.prv
        node.prv = node.nxt = None


cache = LRUCache(1)

cache.put(1, 1)
print cache.get(1)
cache.put(3, 3)
print cache.get(2)
cache.put(4, 4)
print cache.get(1)
print cache.get(3)
print cache.get(4)