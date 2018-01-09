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

    def threeSum(self, nums):
        raise NotImplementedError
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        twoSum = {}
        for i in xrange(len(nums)):
            for j in xrange(i+1, len(nums)):
                twoSum.setdefault(nums[i] + nums[j], []).append((i, j))

        ret = set()
        for k in xrange(len(nums)):
            if -nums[k] in twoSum:
                for i, j in twoSum[-nums[k]]:
                    if k < i and k < j:
                        ret.add(tuple([nums[k], nums[i], nums[j]]))
        return map(list, ret)


    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        h = ListNode(0)
        h.next = head
        prev = h
        guard = prev.next
        for _ in xrange(0, n):
            guard = guard.next
        while guard:
            prev = prev.next
            guard = guard.next
        prev.next = prev.next.next
        return h.next

    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        solutions = [set() for _ in xrange(n+1)]
        solutions[1].add('()')

        for i in xrange(2, n+1):
            for solution in solutions[i-1]:
                solutions[i].add('(' + solution + ')')
            for p in xrange(1, i):
                q = i - p
                for p_sol in solutions[p]:
                    for q_sol in solutions[q]:
                        solutions[i].add(p_sol + q_sol)
        return list(solutions[n])

    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        from heap import Heap

        q = Heap(compare_func=lambda a, b: a.val < b.val)

        for l in lists:
            if l:
                q.insert(l)
        ret = ListNode(0)
        tail = ret

        while not q.is_empty():
            smallest = q.pop_top()
            tail.next = smallest
            tail = tail.next
            if tail.next:
                q.insert(tail.next)
        return ret.next

    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        last_idx = -1
        for i in xrange(len(nums)):
            if last_idx == -1 or nums[i] != nums[last_idx]:
                last_idx += 1
                nums[last_idx] = nums[i]
        return last_idx + 1

    def removeElement(self, nums, val):
        """
        :type nums: List[int]
        :type val: int
        :rtype: int
        """
        last_idx = -1
        for i in xrange(len(nums)):
            if nums[i] != val:
                last_idx += 1
                nums[last_idx] = nums[i]
        print nums[:last_idx+1]
        return last_idx + 1

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """

        def _bin_search(left, right):
            if left >= right:
                return -1
            if left == right-1:
                if nums[left] == target:
                    return left
                else:
                    return -1
            mid = (left + right) / 2
            ret = _bin_search(left, mid)
            if ret != -1:
                return ret
            else:
                return _bin_search(mid, right)

        def _illy_search(left, right):
            if left >= right:
                return -1
            if left == right-1:
                if nums[left] == target:
                    return left
                else:
                    return -1
            mid = (left + right) / 2
            if nums[mid] > nums[left]:
                # left part normal
                ret = _bin_search(left, mid)
                if ret != -1:
                    return ret
                else:
                    return _illy_search(mid, right)
            else:
                # right part normal
                ret = _bin_search(mid, right)
                if ret != -1:
                    return ret
                else:
                    return _illy_search(left, mid)

        return _illy_search(0, len(nums))

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        def findSmallestIndex(left, right):
            print left, right
            if left >= right:
                return -1
            if left == right-1:
                if nums[left] == target:
                    return left
                else:
                    return -1
            mid = (left + right) / 2
            if nums[mid] < target:
                return findSmallestIndex(mid, right)
            elif nums[mid] > target:
                return findSmallestIndex(left, mid)
            else:
                ret = findSmallestIndex(left, mid)
                if ret != -1:
                    return ret
                else:
                    return mid

        def findLargestIndex(left, right):
            if left >= right:
                return -1
            if left == right-1:
                if nums[left] == target:
                    return left
                else:
                    return -1
            mid = (left + right) / 2
            if nums[mid] <= target:
                return findLargestIndex(mid, right)
            elif nums[mid] > target:
                return findLargestIndex(left, mid)

        return [findSmallestIndex(0, len(nums)), findLargestIndex(0, len(nums))]

print Solution().searchRange([5, 7, 7, 8, 8, 10], 8)
