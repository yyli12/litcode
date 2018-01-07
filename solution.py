from linked_list import make_list, ListNode

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        numIndex = {}
        for index, num in enumerate(nums):
            numIndex.setdefault(num, []).append(index)
        for num in nums:
            if target - num in numIndex:
                if target - num == num and len(numIndex[num]) < 2:
                    pass
                else:
                    return [numIndex[num][0], numIndex[target - num][-1]]
        return []

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        ret = ListNode(0)
        curr = ret
        n1 = l1
        n2 = l2
        carry = 0
        while n1 or n2 or carry:
            newval = (n1.val if n1 else 0) + (n2.val if n2 else 0) + carry
            curr.next = ListNode(newval % 10)
            carry = newval / 10
            if n1:
                n1 = n1.next
            if n2:
                n2 = n2.next
            curr = curr.next
        return ret.next

    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s) <= 1:
            return len(s)
        left = right = 0
        maxlen = 0
        charset = set()
        while right < len(s):
            if s[right] not in charset:
                charset.add(s[right])
                if right - left + 1 > maxlen:
                    maxlen = right - left + 1
                right += 1
            else:
                charset.remove(s[left])
                left += 1
        return maxlen

    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        elif x < 0:
            return -self.reverse(-x)
        else:
            ret = 0
            while x:
                ret *= 10
                ret += x % 10
                x /= 10
            if not (-2 ** 32 < ret < 2 ** 31 - 1):
                ret = 0
            return ret

    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        if not strs:
            return None
        minstr = strs[0]
        minlen = len(minstr)
        for str in strs:
            if len(str) < minlen:
                minstr = str
                minlen = len(minstr)
        for i, c in enumerate(minstr):
            for str in strs:
                if str[i] != c:
                    return minstr[:i]
        return minstr

    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        keyboard = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jlk',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz',
        }
        pre = ['']
        ret = []
        for num in digits:
            ret = []
            for char in keyboard[num]:
                for prefix in pre:
                    ret.append(prefix + char)
            pre = ret
        return ret

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s) <= 1:
            return s

        ret = ''

        # odd length
        for center_at in xrange(len(s)):
            for i in xrange(len(s) / 2 + 1):
                l = center_at - i
                r = center_at + i
                if l >= 0 and r < len(s) and s[l] == s[r]:
                    if r - l + 1 > len(ret):
                        ret = s[l:r + 1]
                else:
                    break

        for center_at in xrange(len(s)):
            for i in xrange(len(s) / 2 + 1):
                l = center_at - i
                r = center_at + 1 + i
                if l >= 0 and r < len(s) and s[l] == s[r]:
                    if r - l + 1 > len(ret):
                        ret = s[l:r + 1]
                else:
                    break

        return ret




print Solution().longestPalindrome('abbac')
