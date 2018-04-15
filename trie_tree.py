import pprint

class Trie(object):
    TERM = '#'

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}


    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur[self.TERM] = True


    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        cur = self.root
        for c in word:
            if c not in cur:
                return False
            cur = cur[c]
        return cur.get(self.TERM, False)

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        cur = self.root
        for c in prefix:
            if c not in cur:
                return False
            cur = cur[c]
        return True


class WordDictionary(object):
    TERM = '#'

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """
        cur = self.root
        for c in word:
            if c not in cur:
                cur[c] = {}
            cur = cur[c]
        cur[self.TERM] = True

    def search(self, word, root=None):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        if root is None:
            root = self.root

        if not word:
            return root.get(self.TERM, False)

        c = word[0]
        if c != '.':
            if c in root:
                return self.search(word[1:], root[c])
            else:
                return False
        else:
            return any(self.search(word[1:], root[c]) for c in root)


class WordFilter(object):

    def __init__(self, words):
        """
        :type words: List[str]
        """
        self.prefix = {"#": set()}
        self.suffix = {"#": set()}
        self.weight = {}
        for i, word in enumerate(words):
            pre = self.prefix
            for c in word:
                pre['#'].add(word)
                if c not in pre:
                    pre[c] = {"#": set()}
                pre = pre[c]
            pre['#'].add(word)
            self.weight[word] = i


        for i, word in enumerate(words):
            suf = self.suffix
            for c in word[::-1]:
                suf['#'].add(word)
                if c not in suf:
                    suf[c] = {"#": set()}
                suf = suf[c]
            suf['#'].add(word)


    def f(self, prefix, suffix):
        """
        :type prefix: str
        :type suffix: str
        :rtype: int
        """
        pre = self.prefix
        prematch = pre['#']
        for c in prefix:
            if c not in pre:
                prematch = set()
            else:
                pre = pre[c]
                prematch = pre['#']

        suf = self.suffix
        sufmatch = suf['#']
        for c in suffix[::-1]:
            if c not in suf:
                sufmatch = set()
            else:
                suf = suf[c]
                sufmatch = suf['#']
        intersect = prematch.intersection(sufmatch)

        return max(map(lambda word: self.weight[word], intersect)) if intersect else -1



# class WordFilter(object):
#
#     def __init__(self, words):
#         """
#         :type words: List[str]
#         """
#         from collections import defaultdict
#         self.prefixes = defaultdict(set)
#         self.suffixes = defaultdict(set)
#         for index, word in enumerate(words):
#             prefix, suffix = '', ''
#             for char in [''] + list(word):
#                 prefix += char
#                 self.prefixes[prefix].add(index)
#             for char in [''] + list(word[::-1]):
#                 suffix += char
#                 self.suffixes[suffix[::-1]].add(index)
#
#     def f(self, prefix, suffix):
#         weight = -1
#         for new_w in self.prefixes[prefix] & self.suffixes[suffix]:
#             if new_w > weight:
#                 weight = new_w
#         return weight


def test():
    trie = WordFilter(["cabaabaaaa","ccbcababac","bacaabccba","bcbbcbacaa","abcaccbcaa","accabaccaa","cabcbbbcca","ababccabcb"])
    print trie.f('b', 'a')



if __name__ == '__main__':
    test()