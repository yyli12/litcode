class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

    def __str__(self):
        node = self
        ret = ''
        while node:
            ret += str(node.val) + '->'
            node = node.next
        return ret[:-2]


def make_list(val_arr):
    head = ListNode(0)
    curr = head
    for val in val_arr:
        curr.next = ListNode(val)
        curr = curr.next
    return head.next