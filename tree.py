class MTreeNode(object):
    def __init__(self, x):
        self.val = x
        self.son = []


class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

    def __str__(self):
        return serialize(self)

    def __repr__(self):
        return str(self)


def serialize(root):
    """Encodes a tree to a single string.

    :type root: TreeNode
    :rtype: str
    """
    nodes = []
    nextlevel = [root, ]
    while nextlevel:
        thislevel, nextlevel = nextlevel, []
        for node in thislevel:
            if node is None:
                nodes.append(node)
            else:
                nodes.append(node.val)
                nextlevel.append(node.left)
                nextlevel.append(node.right)
    tail = len(nodes) - 1
    while tail >= 0 and nodes[tail] is None:
        tail -= 1
    return str(nodes[:tail + 1])

def deserialize(data):
    """Decodes your encoded data to tree.

    :type data: str
    :rtype: TreeNode
    """
    nodes = eval(data)
    if not nodes:
        return None

    root = TreeNode(nodes[0])
    queue = [root, ]
    for i in xrange(1, len(nodes)):
        if nodes[i] is None:
            node = None
        else:
            node = TreeNode(nodes[i])
            queue.append(node)
        parent = queue[(i - 1) / 2]
        if i & 1 == 0:
            parent.right = node
        else:
            parent.left = node
    return root
