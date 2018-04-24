# coding=utf-8
from linked_list import make_list, ListNode
from tree import *
from utils import *
from pprint import pprint
import heapq
import bisect

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

    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        import ctypes
        a = ((1 << 32) - 1) & a
        b = ((1 << 32) - 1) & b
        carry_forward = 0
        bit = 0
        ret = 0
        while a or b or carry_forward:
            val = (a & 1) ^ (b & 1) ^ carry_forward
            carry_forward = (a & 1) & (b & 1) | carry_forward & (a & 1) | carry_forward & (b & 1)
            ret |= val << bit
            bit += 1
            a >>= 1
            b >>= 1
        return ctypes.c_int32(((1 << 32) - 1) & ret).value

    def _topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        counter = {}
        for num in nums:
            if num in counter:
                counter[num] += 1
            else:
                counter[num] = 1

        return map(lambda (c, num): num, sorted([(-c, num) for num, c in counter.iteritems()]))[:k]

    def topKFrequent(self, nums, k):
        class minHeap(object):
            def __init__(self):
                self._heap = []

            @property
            def size(self):
                return len(self._heap)

            def bottom_up(self, idx=-1):
                if idx == -1:
                    idx = self.size - 1
                if idx == 0:
                    return
                parent = idx / 2
                v_parent = self._heap[parent]
                v_idx = self._heap[idx]
                if self._heap[parent] > self._heap[idx]:
                    self.swap(parent, idx)
                    self.bottom_up(parent)

            def top_down(self, idx=0):
                if 2 * idx + 1 >= self.size:
                    return
                son1 = 2 * idx + 1
                son2 = 2 * idx + 2
                if son2 >= self.size:
                    son = son1
                else:
                    son = son1 if self._heap[son1] < self._heap[son2] else son2
                if self._heap[idx] > self._heap[son]:
                    self.swap(idx, son)
                    self.top_down(son)

            def add_node(self, node):
                self._heap.append(node)
                self.bottom_up()

            def pop_top(self):
                top = self._heap[0]
                node = self._heap.pop()
                if self.size:
                    self._heap[0] = node
                    self.top_down()
                return top

            def get_top(self):
                return self._heap[0]

            def swap(self, i, j):
                if self._heap[j][1] == -89 or self._heap[i][1] == -89:
                    v_j = self._heap[j]
                    v_i = self._heap[i]
                    pass
                self._heap[i], self._heap[j] = self._heap[j], self._heap[i]

        heap = minHeap()
        counter = {}
        for i in xrange(len(nums)):
            num = nums[i]
            if num in counter:
                counter[num] += 1
            else:
                counter[num] = 1

        for num, count in counter.iteritems():
            if num == 89:
                pass
            if heap.size < k:
                heap.add_node((count, -num))
            else:
                min_count, min_num = heap.get_top()
                if count > min_count:
                    heap._heap[0] = (count, -num)
                    heap.top_down()

        ret = []
        print heap._heap
        while heap.size:
            top = heap.pop_top()
            print top
            ret.append(-top[1])
        return ret[::-1]

    def getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """
        MAX_BIT = 2 ** 32
        MAX_BIT_COMPLIMENT = -2 ** 32

        while b != 0:

            if b == MAX_BIT:
                return a ^ MAX_BIT_COMPLIMENT

            carry = a & b
            print carry

            a = a ^ b
            print a

            b = carry << 1
            print b

        print a
        return a

    def _getSum(self, a, b):
        """
        :type a: int
        :type b: int
        :rtype: int
        """

        def plus1(x):
            plus = 1
            while x & 1:
                x ^= plus
                plus <<= 1
            x ^= plus
            return x

        import ctypes
        a = 0xffffffff & a
        b = 0xffffffff & b
        carry_forward = 0
        bit = 0
        ret = 0
        while a or b or carry_forward:
            val = (a & 1) ^ (b & 1) ^ carry_forward
            carry_forward = (a & 1) & (b & 1) | carry_forward & (a & 1) | carry_forward & (b & 1)
            ret |= val << bit
            before = bit
            bit = plus1(bit)
            print before, bit
            a >>= 1
            b >>= 1
        return ctypes.c_int32(0xffffffff & ret).value

    def inorderTraversal(self, root):

        stack = []
        node = root
        while node:
            stack.append(node)
            node = node.left

        ret = []
        while stack:
            node = stack.pop()
            ret.append(node.val)
            node = node.right
            while node:
                stack.append(node)
                node = node.left

        return ret

    def titleToNumber(self, s):
        """
        :type s: str
        :rtype: int
        """
        def charToNumber(c):
            return ord(c) - ord('A') + 1

        ret = 0
        for c in s:
            ret = ret * 26 + charToNumber(c)
        return ret

    def convertToTitle(self, n):
        """
        :type n: int
        :rtype: str
        """
        def numberToChar(n):
            return 'ZABCDEFGHIJKLMNOPQRSTUVWXY'[n]

        ret = ''

        while n > 0:
            ret += numberToChar(n % 26)
            n = (n - 1) / 26

        return ret

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        MAPPING = {
            'I': 1,
            'V': 5,
            'X': 10,
            'L': 50,
            'C': 100,
            'D': 500,
            'M': 1000,
        }

        def _getLargest(s):
            return sorted(s, key=lambda c: MAPPING[c], reverse=True)[0]


        def _romanToInt(s):
            if not s:
                return 0

            l = _getLargest(s)

            start_index = 0
            while start_index < len(s):
                if s[start_index] == l:
                    end_index = start_index
                    while end_index < len(s) and s[end_index] == l:
                        end_index += 1
                    break
                start_index += 1
            return MAPPING[l] * (end_index - start_index) - _romanToInt(s[:start_index]) + _romanToInt(s[end_index:])

        return _romanToInt(s)

    def maxProfit2(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        profit = 0
        for day in xrange(1, len(prices)):
            daily_profit = prices[day] - prices[day-1]
            profit += daily_profit if daily_profit > 0 else 0
        return profit

    def maxProfit1(self, prices):
        lowest = prices[0]
        profit = 0
        for price in prices:
            if price < lowest:
                lowest = price
            if price - lowest > profit:
                profit = price - lowest
        return profit

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        start = range(len(nums))
        lsum = list(nums)
        print start, lsum
        for index in xrange(1, len(nums)):
            if lsum[index - 1] + nums[index] > lsum[index]:
                start[index] = start[index - 1]
                lsum[index] = lsum[index - 1] + nums[index]

        return max(lsum)

    def frequencySort(self, s):
        """
        :type s: str
        :rtype: str
        """
        counter = {}
        for c in s:
            if c in counter:
                counter[c] += 1
            else:
                counter[c] = 1

        return ''.join(c * count for count, c in sorted(((count, c) for c, count in counter.iteritems()), reverse=True))

    def _permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def _p(nums):
            if len(nums) == 1:
                return [nums, ]
            ret = []
            for index in xrange(len(nums)):
                for x in _p(nums[:index] + nums[index + 1:]):
                    ret.append([nums[index], ] + x)
            return ret

        return _p(nums)

    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        helper = [[]]
        res = list()
        for n in nums:
            res = list()
            for h in helper:
                for i in range(len(h)+1):
                    s = h[:]
                    s.insert(i, n)
                    res.append(s)
            helper = res[:]
            print helper
        return res

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        def _recusive(head):
            if head.next is None:
                return head, head
            n, next = head, head.next
            n.next = None
            h, t = _recusive(next)
            t.next = n
            return h, n
        if head is None:
            return head
        return _recusive(head)[0]


        # non-recursive
        # h = ListNode(0)
        # ret = ListNode(0)
        # h.next = head
        # while h.next:
        #     n, h.next = h.next, h.next.next
        #     ret.next, n.next = n, ret.next
        # return ret.next


    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        count = 0

        l = ListNode(0)
        l.next = head
        ret = ListNode(0)
        h = ret
        t = ret
        while l.next:
            count += 1
            c, l.next = l.next, l.next.next
            c.next = None
            if count < m:
                h.next = c
                h = h.next
            elif m <= count <= n:
                c.next, h.next = h.next, c
                if count == m:
                    t = c
            elif count > n:
                t.next = c
                t = t.next
        return ret.next

    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        def count(root):
            if not root:
                return 0
            if hasattr(root, ''):
                return root.c
            c = count(root.left) + count(root.right) + 1
            root.c = c
            return c

        def findK(root, k):
            smaller = count(root.left)
            if smaller >= k:
                return findK(root.left, k)
            elif smaller == k - 1:
                return root.val
            else:
                return findK(root.right. k - smaller)

        return findK(root, k)

    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.append(-1)
        for index in xrange(len(nums)):
            if nums[index] == -1:
                continue
            while index != nums[index] and nums[index] != -1:
                nums[nums[index]], nums[index] = nums[index], nums[nums[index]]
        for index in xrange(len(nums)):
            if nums[index] == -1:
                return index

    def oddEvenList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        h = ListNode(0)
        h.next = head
        h1 = ListNode(0)
        h2 = ListNode(0)
        count = 0
        t_odd = h1
        t_even = h2
        while h.next:
            n, h.next = h.next, h.next.next
            n.next = None
            count += 1

            if count & 1:
                # odd
                n.next, t_odd.next = t_odd.next, n
                t_odd = n
            else:
                # even
                n.next, t_even.next = t_even.next, n
                t_even = n
        t_odd.next = h2.next
        return h1.next

    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        nums1.sort()
        nums2.sort()
        ret = []
        i1 = 0
        i2 = 0
        while i1 < len(nums1) and i2 < len(nums2):
            if nums1[i1] == nums2[i2]:
                ret.append(nums1[i1])
                i1 += 1
                i2 += 1
            elif nums1[i1] > nums2[i2]:
                i2 += 1
            else:
                i1 += 1
        return ret

    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        ret = [[]]
        for num in nums:
            new = []
            for s in ret:
                new.append(s + [num, ])
            ret += new

        return ret

    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if not nums:
            return None
        mid = len(nums) / 2
        root = TreeNode(nums[mid])
        root.left = self.sortedArrayToBST(nums[:mid])
        root.right = self.sortedArrayToBST(nums[mid+1:])
        return root

    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if not head:
            return None
        if not head.next:
            return TreeNode(head.val)
        one = head
        two = head.next
        while two.next and two.next.next:
            one = one.next
            two = two.next.next
        node = one.next
        one.next = None
        second_half = node.next
        node.next = None

        root = TreeNode(node.val)
        root.left = self.sortedListToBST(head)
        root.right = self.sortedListToBST(second_half)
        return root

    def decodeString(self, s):
        """
        :type s: str
        :rtype: str
        """

        # recursive
        class timeString():

            def __init__(self, string):
                i = 0
                while string[i] != '[':
                    i += 1
                self._count = int(string[:i])
                self._inside = string[i+1:-1]
                self._parsed = []
                self._parse()

            def _parse(self):
                string = ''
                begin = None
                self._parsed = []
                for i, c in enumerate(self._inside):
                    if c.isdigit():
                        if begin is None:
                            begin = i
                            stack = 0
                            if string:
                                self._parsed.append(string)
                    elif c == '[':
                        stack += 1
                    elif c == ']':
                        stack -= 1
                        if stack == 0:
                            self._parsed.append(self._inside[begin:i+1])
                            string = ''
                            begin = None
                    else:
                        string += c

                if string:
                    self._parsed.append(string)

            def __str__(self):
                return ''.join(map(lambda part: str(timeString(part)) if part[0].isdigit() else part, self._parsed)) * self._count

        """
        :type s: str
        :rtype: str
        """

        # non recursive
        s = '1[%s]' % s

        count_stack = []
        string_stack = []

        count = 0
        string = ''
        digit = False
        for c in s:
            if c.isdigit():
                if not digit:
                    string_stack.append(string)
                    string = ''
                    digit = True
                count = count * 10 + int(c)
            elif c == '[':
                digit = False
                count_stack.append(count)
                count = 0
            elif c == ']':
                string = string * count_stack.pop()
                if string_stack:
                    prefix = string_stack.pop()
                    string = prefix + string

            else:
                string += c
        return string

    @timeit
    def findTheDigits(self, n, d):

        the_digits_count = 1
        while d >= 10 ** the_digits_count:
            the_digits_count += 1

        ret = [False] * n
        higher_digits = 0 if d != 0 else 1
        while True:
            existing_higher = False
            lower_digit_count = 0
            while True:
                the_digits = d * 10 ** lower_digit_count
                existing_lower = False
                for lower_digits in xrange(10 ** lower_digit_count):
                    whole_number = higher_digits * 10 ** (the_digits_count + lower_digit_count) + the_digits + lower_digits
                    if whole_number < n:
                        existing_lower = True
                        existing_higher = True
                        ret[whole_number] = True
                lower_digit_count += 1
                if not existing_lower:
                    break
            higher_digits += 1
            if not existing_higher:
                break

        return ret

    @timeit
    def findTheDigitsSlow(self, n, d):
        the_digits_count = 1
        while d >= 10 ** the_digits_count:
            the_digits_count += 1

        divisor = 10 ** the_digits_count
        ret = [False] * n
        for num in xrange(n):
            origin_num = num
            while num:
                if num % divisor == d:
                    ret[origin_num] = True
                    break
                num /= 10
        return ret

    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        count = [[0] * n for _ in xrange(m)]

        count[m - 1][n - 1] = 1
        for row in xrange(m - 1, -1, -1):
            for col in xrange(n - 1, -1, -1):
                print row, col
                if row + 1 < m:
                    count[row][col] += count[row + 1][col]
                if col + 1 < n:
                    count[row][col] += count[row][col + 1]
        return count[0][0]

    def calculateMinimumHP(self, dungeon):
        m, n = len(dungeon), len(dungeon[0])
        minHP = [[0] * n for _ in xrange(m)]

        minHP[m - 1][n - 1] = max(1 - dungeon[m - 1][n - 1], 1)
        print minHP
        for row in xrange(m - 1, -1, -1):
            for col in xrange(n - 1, -1, -1):
                minnext = None
                if row + 1 < m:
                    minnext = dungeon[row+1][col]
                if col + 1 < n:
                    if minnext is None:
                        minnext = dungeon[row][col+1]
                    else:
                        minnext = min(minnext, dungeon[row][col+1])
                if minnext is None:
                    continue
                minHP[row][col] = max(minnext - dungeon[row][col], 1)
        return minHP[0][0]

    def cherryPickup(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        n = len(grid)
        maxs = [[0] * n for _ in xrange(n)]
        directions = [[None] * n for _ in xrange(n)]

        maxs[n - 1][n - 1] = grid[n - 1][n - 1]
        directions[n - 1][n - 1] = (0, 0)
        for row in xrange(n - 1, -1, -1):
            for col in xrange(n - 1, -1, -1):
                if grid[row][col] == -1 or (row, col) == (n, n):
                    continue
                maxnext = None
                direction = None
                if row + 1 < n and directions[row + 1][col] is not None:
                    maxnext = maxs[row + 1][col]
                    direction = (1, 0)
                if col + 1 < n and directions[row][col + 1] is not None:
                    if maxnext is None:
                        maxnext = maxs[row][col + 1]
                        direction = (0, 1)
                    else:
                        if maxs[row][col + 1] > maxnext:
                            maxnext = maxs[row][col + 1]
                            direction = (0, 1)
                if maxnext is not None:
                    maxs[row][col] = maxnext + grid[row][col]
                    directions[row][col] = direction

        path = []
        cur = (0, 0)
        while directions[cur[0]][cur[1]] and cur != (n - 1, n - 1):
            path.append(cur)
            direction = directions[cur[0]][cur[1]]
            cur = (cur[0] + direction[0], cur[1] + direction[1])
        if cur == (n - 1, n - 1):
            path.append(cur)
        else:
            return 0

        for cur in path:
            grid[cur[0]][cur[1]] = 0

        maxs2 = [[0] * n for _ in xrange(n)]
        for row in xrange(n - 1, -1, -1):
            for col in xrange(n - 1, -1, -1):
                if grid[row][col] == -1 or (row, col) == (n, n):
                    continue
                maxnext = None
                if row + 1 < n and grid[row + 1][col] != -1:
                    maxnext = maxs2[row + 1][col]
                if col + 1 < n and grid[row][col + 1] != -1:
                    if maxnext is None:
                        maxnext = maxs2[row][col + 1]
                    else:
                        if maxs2[row][col + 1] > maxnext:
                            maxnext = maxs2[row][col + 1]
                if maxnext is not None:
                    maxs2[row][col] = maxnext + grid[row][col]

        return maxs[0][0] + maxs2[0][0]

    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        n = len(matrix)
        cn = 1.0 * (n - 1) / 2

        def getRotatedRowCol(r, c):
            return int(c), int(2 * cn - r)

        for r0 in xrange(int(cn + 1) if n % 2 == 0 else int(cn)):
            for c0 in xrange(int(cn + 1)):
                r1, c1 = getRotatedRowCol(r0, c0)
                r2, c2 = getRotatedRowCol(r1, c1)
                r3, c3 = getRotatedRowCol(r2, c2)
                matrix[r0][c0], matrix[r1][c1], matrix[r2][c2], matrix[r3][c3] = \
                matrix[r3][c3], matrix[r0][c0], matrix[r1][c1], matrix[r2][c2]

    def climbStairs(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n <= 1:
            return 1
        steps = [1] * (n + 1)
        for i in xrange(2, n + 1):
            steps[i] = steps[i - 1] + steps[i - 2]
        return steps[n]

    def minCostClimbingStairs(self, cost):
        """
        :type cost: List[int]
        :rtype: int
        """
        if len(cost) < 2:
            return min(cost)
        n = len(cost)
        mincost = [None] * (n + 1)
        mincost[0] = 0
        mincost[1] = 0
        for i in xrange(2, n + 1):
            mincost[i] = min(mincost[i - 1] + cost[i - 1], mincost[i - 2] + cost[i - 2])
        return mincost[-1]

    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        visited = set()
        nextnum = n
        while nextnum != 1 and nextnum not in visited:
            num = nextnum
            visited.add(num)
            nextnum = 0
            while num:
                nextnum += (num % 10) ** 2
                num /= 10
        return nextnum == 1

    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """

        for divisor in [2, 3, 5]:
            while num % divisor == 0:
                num /= divisor

        return num == 1

    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        nums = [1]
        nexts = [2 << 31, 2 << 31, 2, 3, 2 << 31, 5]
        idx = [0, 0, 0, 0, 0, 0]
        while len(nums) < n:
            factor = nexts.index(min(nexts))
            if nums[-1] != nexts[factor]:
                nums.append(nexts[factor])
            idx[factor] += 1
            nexts[factor] = nums[idx[factor]] * factor
        return nums[-1]

    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        def _add_nums(nums):
            maxlen = max(map(len, nums))
            carry = 0
            ds = []
            for i in xrange(maxlen):
                n = carry
                for num in nums:
                    if i < len(num):
                        n += num[i]
                ds.append(n % 10)
                carry = n / 10
            while carry:
                ds.append(carry % 10)
                carry /= 10
            ds = ds[::-1]
            non_zero = 0
            while non_zero < len(ds) and ds[non_zero] == 0:
                non_zero += 1
            ds = ds[non_zero:]
            if not ds:
                return '0'
            return ''.join(map(str, ds))

        def _digit_multi(d1, num, zeros):
            ret = [0] * zeros
            carry = 0
            for d2 in num:
                n = (ord(d2) - ord('0')) * (ord(d1) - ord('0')) + carry
                ret.append(n % 10)
                carry = n / 10
            while carry:
                ret.append(carry % 10)
                carry /= 10
            return ret
        if num1 < num2:
            num1, num2 = num1, num2
        n1 = list(num1[::-1])
        n2 = list(num2[::-1])
        addons = []
        for i in xrange(len(n1)):
            addons.append(_digit_multi(n1[i], n2, i))
        return _add_nums(addons)

    def generate(self, numRows):
        """
        :type numRows: int
        :rtype: List[List[int]]
        """
        ret = []
        for i in xrange(numRows):
            if i == 0:
                ret.append([1])
            else:
                row = []
                for j in xrange(i + 1):
                    if j == 0 or j == i:
                        row.append(1)
                    else:
                        row.append(ret[i-1][j-1] + ret[i-1][j])
                ret.append(row)
        return ret

    def findMin(self, nums):
        """
        :type nums: List[int] - an array sorted in ascending order is rotated at some pivot
        :rtype: int
        """
        def _find(nums):
            if len(nums) <= 2:
                return min(nums)
            midi = len(nums) / 2
            mid = nums[midi]
            if nums[0] <= mid <= nums[-1]:
                return nums[0]
            print nums[:midi+1]
            if nums[0] > mid:
                return _find(nums[:midi+1])
            else:
                return _find(nums[midi:])

        return _find(nums)

    def increasingTriplet(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        if len(nums) < 3:
            return False
        small1 = 1 << 32
        small2 = 1 << 32
        for num in nums:
            print small1, small2, num
            if num <= small1:
                small1 = num
            elif num <= small2:
                small2 = num
            else:
                return True
        return False

    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        ra = a[::-1]
        rb = b[::-1]
        rret = ''

        carry = 0
        for i in xrange(max(len(ra), len(rb))):
            d = carry
            if i < len(ra):
                d += int(ra[i])
            if i < len(rb):
                d += int(rb[i])
            carry = (d & 2) >> 1
            d &= 1
            rret += str(d)
        if carry:
            rret += str(carry)

        return rret[::-1]

    def largestDivisibleSubset(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        def get_longest(son_nodes, i):
            def _get_longest(i):
                if i not in memo:
                    if i not in son_nodes or not son_nodes[i]:
                        memo[i] = [i, ]
                    else:
                        memo[i] = [i, ] + sorted((_get_longest(s) for s in son_nodes[i]), key=len, reverse=True)[0]
                return memo[i]
            memo = {}
            return _get_longest(i)

        nums.sort()
        has_one = nums[0] == 1
        if has_one:
            nums = nums[1:]
        son_nodes = {
            1: set(),
        }
        for num in nums:
            direct_ancestors = {1, }
            changed = True
            while changed:
                changed = False
                next_level_ancestors = set()
                for ancestor in direct_ancestors:
                    no_next_level_son = True
                    if ancestor in son_nodes:
                        for son in son_nodes[ancestor]:
                            if num % son == 0:
                                next_level_ancestors.add(son)
                                changed = True
                                no_next_level_son = False
                    if no_next_level_son:
                        next_level_ancestors.add(ancestor)
                direct_ancestors = next_level_ancestors

            for ancestor in direct_ancestors:
                son_nodes.setdefault(ancestor, set()).add(num)

        if has_one:
            return get_longest(son_nodes, 1)
        else:
            return get_longest(son_nodes, 1)[1:]

    def maxIncreaseKeepingSkyline(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0

        rowmax = map(max, grid)
        colmax = map(max, ([row[col] for row in grid] for col in xrange(len(grid[0]))))

        ret = 0
        for row in xrange(len(grid)):
            for col in xrange(len(grid[0])):
                ret += grid[row][col] - min(rowmax[row], colmax[col])
        return ret

    def findCircleNum(self, M):
        """
        :type M: List[List[int]]
        :rtype: int
        """
        def disjointSet(M):
            def get_root(i):
                if groups[i] != i:
                    groups[i] = get_root(groups[i])
                return groups[i]

            n = len(M)
            groups = [i for i in xrange(n)]

            for i in xrange(n):
                for j in xrange(i):
                    if M[i][j]:
                        groups[get_root(j)] = get_root(i)
            return len(set(get_root(i) for i in xrange(n)))

        def graph(M):
            visited = set()
            n = len(M)
            r = 0

            for i in xrange(n):
                if i not in visited:
                    r += 1
                    l = [i, ]
                    while l:
                        cur = l.pop(0)
                        visited.add(cur)
                        for nb in filter(lambda x: M[cur][x] == 1, xrange(n)):
                            if nb not in visited:
                                l.append(nb)

            return r

        return graph(M)

    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        def findKth(nums1, nums2):
            pass

    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """

        n = len(nums)
        if n <= 1:
            return n

        lis_len = [1] * n

        for i in xrange(n):
            max_len = 0
            for j in xrange(i - 1, -1, -1):
                if nums[j] < nums[i] and lis_len[j] > max_len:
                    max_len = lis_len[j]
                lis_len[i] = 1 + max_len
        return max(lis_len)

    def twoSum2(self, numbers, target):
        """
        :type numbers: List[int]
        :type target: int
        :rtype: List[int]
        """
        n = len(numbers)
        if n <= 2:
            return
        i, j = 0, n - 1
        s = numbers[i] + numbers[j]
        while s != target:
            if s > target:
                j -= 1
            else:
                i += 1
            s = numbers[i] + numbers[j]
        return [i+1, j+1]

    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """

        def sol1(matrix, target):
            m = len(matrix)
            if m <= 0:
                return False
            n = len(matrix[0])
            if n <= 0:
                return False

            r, c = 0, n - 1

            while r < m and c > 0:
                print matrix[r][c]
                if matrix[r][c] == target:
                    return True
                elif matrix[r][c] < target:
                    # too small -> not in this row
                    r += 1
                else:
                    # too large -> not in this col
                    c -= 1
            return False

        def sol2(matrix, target):
            m = len(matrix)
            if m <= 0:
                return False
            n = len(matrix[0])
            if n <= 0:
                return False

            q = [(m / 2, n / 2), ]
            i = 0
            visited = set()
            while i < len(q):
                r, c = q[i]
                if 0 <= r < m and 0 <= c < n and (r, c) not in visited:
                    visited.add((r, c))
                    v = matrix[r][c]
                    if v == target:
                        return True
                    elif v > target:
                        q.append((r - 1, c))
                        q.append((r, c - 1))
                    else:
                        q.append((r + 1, c))
                        q.append((r, c + 1))
                i += 1
            return False

        return sol2(matrix, target)

    def leet581(self, nums):

        n = len(nums)
        if n <= 0:
            return 0

        sorted_nums = sorted(nums)
        l = 0
        r = n - 1
        while l < n and nums[l] == sorted_nums[l]:
            l += 1
        while r >= 0 and nums[r] == sorted_nums[r]:
            r -= 1
        return r - l + 1

    def rob2(self, nums):
        n = len(nums)
        if n <= 0:
            return 0
        if n <= 3:
            # only rob one
            return max(nums)

        def easyrob(nums):
            n = len(nums)
            if n <= 0:
                return 0
            if n <= 2:
                return max(nums)
            max_with = [0] * n
            max_wout = [0] * n
            max_with[0] = nums[0]

            for i in xrange(1, n):
                max_with[i] = max_wout[i-1] + nums[i]
                max_wout[i] = max(max_with[i-1], max_wout[i-1])

            return max(max(max_wout), max(max_with))

        return max(nums[0] + easyrob(nums[2:-1]), easyrob(nums[1:]))

    def rob3(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """

        def rob(root):
            if not root:
                return 0, 0
            else:
                w_left, wo_left = rob(root.left)
                w_rite, wo_rite = rob(root.right)
                return root.val + wo_left + wo_rite, max(w_left, wo_left) + max(w_rite, wo_rite)

        return max(rob(root))


    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """

        def sol1_heap(nums, k):
            class Heap(object):
                def __init__(self):
                    self.heap = []

                @property
                def size(self):
                    return len(self.heap)

                def insert(self, val):
                    self.heap.append(val)
                    self.go_up(len(self.heap) - 1)

                def get_top(self):
                    return self.heap[0]

                def pop_top(self):
                    top = self.heap[0]
                    self.heap[0] = self.heap[-1]
                    self.heap.pop()
                    if self.heap:
                        self.go_down(0)
                    return top

                def replace_top(self, val):
                    self.heap[0] = val
                    self.go_down(0)

                def go_up(self, i):
                    parent = (i - 1) / 2
                    if parent < 0 or self.heap[parent] < self.heap[i]:
                        return
                    self.heap[parent], self.heap[i] = self.heap[i], self.heap[parent]
                    self.go_up(parent)

                def go_down(self, i):
                    sson = i
                    s1 = 2 * i + 1
                    if s1 < len(self.heap):
                        if self.heap[sson] > self.heap[s1]:
                            sson = s1
                        s2 = 2 * i + 2
                        if s2 < len(self.heap):
                            if self.heap[sson] > self.heap[s2]:
                                sson = s2
                    if sson != i:
                        self.heap[sson], self.heap[i] = self.heap[i], self.heap[sson]
                        self.go_down(sson)

            h = Heap()
            for num in nums:
                if h.size < k:
                    h.insert(num)
                else:
                    if num > h.get_top():
                        h.replace_top(num)

            return h.get_top()

        def sol2_recursion(nums, k):
            def find(nums, k):
                # s s s p p p l l l
                #
                #         ----k----
                pivot = nums[len(nums) / 2]
                smaller = []
                larger = []
                equal = 0
                for num in nums:
                    if num < pivot:
                        smaller.append(num)
                    elif num > pivot:
                        larger.append(num)
                    else:
                        equal += 1
                if len(larger) + 1 <= k <= len(larger) + equal:
                    return pivot
                elif k <= len(larger):
                    return find(larger, k)
                else:
                    return find(smaller, k - (len(larger) + equal))
            return find(nums, k)

    def topKFrequent2(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """

        class Heap(object):
            def __init__(self):
                self.heap = []

            @property
            def size(self):
                return len(self.heap)

            def insert(self, val):
                self.heap.append(val)
                self.go_up(len(self.heap) - 1)

            def get_top(self):
                return self.heap[0]

            def pop_top(self):
                top = self.heap[0]
                self.heap[0] = self.heap[-1]
                self.heap.pop()
                if self.heap:
                    self.go_down(0)
                return top

            def replace_top(self, val):
                self.heap[0] = val
                self.go_down(0)

            def go_up(self, i):
                parent = (i - 1) / 2
                if parent < 0 or self.heap[parent] < self.heap[i]:
                    return
                self.heap[parent], self.heap[i] = self.heap[i], self.heap[parent]
                self.go_up(parent)

            def go_down(self, i):
                sson = i
                s1 = 2 * i + 1
                if s1 < len(self.heap):
                    if self.heap[sson] > self.heap[s1]:
                        sson = s1
                    s2 = 2 * i + 2
                    if s2 < len(self.heap):
                        if self.heap[sson] > self.heap[s2]:
                            sson = s2
                if sson != i:
                    self.heap[sson], self.heap[i] = self.heap[i], self.heap[sson]
                    self.go_down(sson)

        count = {}
        for num in nums:
            if num in count:
                count[num] += 1
            else:
                count[num] = 1

        h = Heap()
        for num, c in count.iteritems():
            if h.size < k:
                h.insert((c, num))
            else:
                if c > h.get_top()[0]:
                    h.replace_top((c, num))

        ret = []
        while h.size:
            ret.append(h.pop_top()[1])
        return ret

    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums) - 1
        low = 1
        high = n + 1
        while high - low > 1:
            # candidate inside [low, high)
            # 1, 2, 3, 4, 4, 5
            candidate = (low + high) / 2
            print low, high, candidate
            smaller = 0
            equal = 0
            for num in nums:
                if num < candidate:
                    smaller += 1
                if num == candidate:
                    equal += 1
                    if equal > 1:
                        return candidate
            print smaller, candidate
            if smaller > candidate - 1:
                # new range: [low, candidate)
                high = candidate
            else:
                # new range: [candidate + 1, high)
                low = candidate + 1
        return low

    def firstMissingPositive(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 1
        nums.append(nums[0])
        i = 1
        while i < len(nums):
            while 0 < nums[i] < len(nums) and i != nums[i] and nums[nums[i]] != nums[i]:
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
            i += 1
        i = 1
        while i < len(nums) and nums[i] == i:
            i += 1
        return i

    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        n = len(nums)
        nums.append(nums[0])
        for i in xrange(1, n+1):
            while i != nums[i] and nums[i] != nums[nums[i]]:
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]

        return filter(lambda i: nums[i] != i, xrange(1, n+1))

    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        n = len(nums)
        nums.append(nums[0])

        ret = set()
        for i in xrange(1, n + 1):
            while i != nums[i] and nums[i] != nums[nums[i]]:
                nums[nums[i]], nums[i] = nums[i], nums[nums[i]]
            if i != nums[i] and nums[i] == nums[nums[i]]:
                ret.add(nums[i])
        return list(ret)

    def minSwapsCouples(self, row):
        """
        :type row: List[int]
        :rtype: int
        """

        def areCouple(p, q):
            return p / 2 == q / 2

        def getCouple(p):
            return p - 1 if p & 1 else p + 1

        n = len(row)
        neighbors = {}
        for i in xrange(0, n, 2):
            neighbors[row[i]] = row[i + 1]
            neighbors[row[i + 1]] = row[i]

        ret = 0
        checked = set()
        for i in xrange(0, n, 2):
            if row[i] in checked:
                continue
            checked.add(row[i])
            checked.add(row[i+1])
            couples = 1
            seed = row[i]
            this_half = row[i+1]
            close_loop = areCouple(seed, this_half)
            while not close_loop:
                other_half = getCouple(this_half)
                this_half = neighbors[other_half]
                checked.add(other_half)
                checked.add(this_half)
                couples += 1
                close_loop = areCouple(seed, this_half)
            ret += couples - 1
        return ret

    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        def sol1(n):
            dp = range(n + 1)
            for num in xrange(1, n+1):
                i = 1
                while i * i <= num:
                    dp[num] = min(dp[num], 1 + dp[num - i * i])
                    i += 1
            return dp[n]

        def sol2(n):
            dp = [0]
            while len(dp) <= n + 1:
                num = len(dp)
                cnt = num
                i = 1
                while i * i <= num:
                    cnt = min(cnt, 1 + dp[num - i * i])
                    i += 1
                dp.append(cnt)
            return dp[n]

        return sol2(n)

    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        check_points = []
        cross1 = []
        cross2 = []
        for i in xrange(9):
            check_points.append((board[i], 'r%s' % i))  # rows
            check_points.append(([row[i] for row in board], 'c%s' % i))  # cols

            blocks = []
            r, c = i / 3 * 3, i % 3 * 3
            for delta_r in xrange(3):
                for delta_c in xrange(3):
                    blocks.append(board[r + delta_r][c + delta_c])
            check_points.append((blocks, 'b%s' % i))  # block

            # make cross
            cross1.append(board[i][i])
            cross2.append(board[8 - i][i])

        check_points.append((cross1, 'x1'))
        check_points.append((cross2, 'x2'))

        pprint(board)

        def check((nums, name)):
            shown = set()
            for num in nums:
                if num in shown:
                    print nums, name
                    return False
                elif ord('0') <= ord(num) <= ord('9'):
                    shown.add(num)
            return True

        return all(check(check_point) for check_point in check_points)

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        rows = [set() for i in xrange(9)]
        cols = [set() for i in xrange(9)]
        blks = [set() for i in xrange(9)]

        def add(r, c, num):
            rows[r].add(num)
            cols[c].add(num)
            blks[r / 3 * 3 + c / 3].add(num)

        def remove(r, c, num):
            rows[r].remove(num)
            cols[c].remove(num)
            blks[r / 3 * 3 + c / 3].remove(num)

        for i, row in enumerate(board):
            for j, num in enumerate(row):
                if ord('0') <= ord(num) <= ord('9'):
                    add(i, j, int(num))

        set9 = set(xrange(1, 10))
        pprint(board)
        def solve():
            options = None
            r, c = None, None
            for i in xrange(9):
                for j in xrange(9):
                    if board[i][j] == '.':
                        new_options = set9 - (rows[i].union(cols[j]).union(blks[i / 3 * 3 + j / 3]))
                        if options is None or len(new_options) < len(options):
                            options = new_options
                            r, c = i, j
            if options is None:
                return True

            for option in options:
                board[r][c] = str(option)
                add(r, c, option)
                if solve():
                    return True
                remove(r, c, option)
                board[r][c] = '.'
            return False

        solve()
        pprint(board)

    def longestIncreasingPath(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: int
        """
        if not matrix or not matrix[0]:
            return 0

        vals = []
        for r, row in enumerate(matrix):
            for c, val in enumerate(row):
                vals.append((val, r, c))

        vals.sort()

        m, n = len(matrix), len(matrix[0])
        dp = [[1] * n for _ in xrange(m)]

        for val, r, c in vals:
            prefix = 0
            if r - 1 >= 0 and matrix[r - 1][c] < val and dp[r - 1][c] > prefix:
                prefix = dp[r - 1][c]
            if c - 1 >= 0 and matrix[r][c - 1] < val and dp[r][c - 1] > prefix:
                prefix = dp[r][c - 1]
            if r + 1 <  m and matrix[r + 1][c] < val and dp[r + 1][c] > prefix:
                prefix = dp[r + 1][c]
            if c + 1 <  n and matrix[r][c + 1] < val and dp[r][c + 1] > prefix:
                prefix = dp[r][c + 1]
            dp[r][c] = prefix + 1

        return max(map(max, dp))

    def trap(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        if not height:
            return 0

        def easytrap(height):
            # guarantee rightmost has maximum height
            print height
            if len(height) <= 1:
                return 0

            ret = 0

            left = 0
            while left != len(height) - 1:
                right = left + 1
                while height[right] < height[left]:
                    ret += height[left] - height[right]
                    right += 1
                left = right

            return ret

        peak = max((h, i) for i, h in enumerate(height))[1]

        return easytrap(height[:peak+1]) + easytrap(height[peak:][::-1])

    def gameOfLife(self, board):
        """
        :type board: List[List[int]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return

        m, n = len(board), len(board[0])

        def count(i, j):
            lives = 0
            deaths = 0
            neighbors = [
                (i - 1, j),
                (i + 1, j),
                (i, j - 1),
                (i, j + 1),
                (i - 1, j - 1),
                (i - 1, j + 1),
                (i + 1, j - 1),
                (i + 1, j + 1),
            ]
            for nr, nc in neighbors:
                if 0 <= nr < m and 0 <= nc < n:
                    lives += board[nr][nc] & 1
                    deaths += 1 - board[nr][nc] & 1
            return lives, deaths

        for r in xrange(m):
            for c in xrange(n):
                lives, deaths = count(r, c)
                if board[r][c]:
                    if not (lives != 2 and lives != 3):
                        board[r][c] |= 1 << 1
                else:
                    if lives == 3:
                        board[r][c] |= 1 << 1

        for r in xrange(m):
            for c in xrange(n):
                board[r][c] >>= 1

    def trailingZeroes(self, n):
        """
        :type n: int
        :rtype: int
        """
        factor = 5
        ret = []
        while factor <= n:
            ret.append(n / factor)
            factor *= 5
        return ret, sum(ret)

    def preimageSizeFZF(self, K):
        """
        :type K: int
        :rtype: int
        """
        def zeros(n):
            factor = 5
            ret = 0
            while factor <= n:
                ret += n / factor
                factor *= 5
            return ret

        left = 1
        right = K * 5
        while left < right:
            mid = (left + right) / 2
            if zeros(mid) < K:
                left = mid + 1
            else:
                right = mid

        if zeros(left) != K:
            return 0
        return 5


    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """

        i = 2
        prev = '1'
        while i <= n:
            nxt = ''
            char = prev[0]
            cnt = 1
            for c in prev[1:]:
                if c == char:
                    cnt += 1
                else:
                    nxt += '%s%s' % (cnt, char)
                    char = c
                    cnt = 1
            nxt += '%s%s' % (cnt, char)
            prev = nxt
            i += 1
        return prev

    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        nextlevel = [root, ]
        level = 0
        ret = []
        while True:
            if not nextlevel:
                break
            ret.append([])
            currlevel = nextlevel
            nextlevel = []

            while currlevel:
                node = currlevel.pop()
                ret[level].append(node.val)
                if level & 1 == 0:
                    if node.left:
                        nextlevel.append(node.left)
                    if node.right:
                        nextlevel.append(node.right)
                else:
                    if node.right:
                        nextlevel.append(node.right)
                    if node.left:
                        nextlevel.append(node.left)
            level += 1

        return ret

    def connect(self, root):
        def make_next(root, sibling):
            if not root:
                return
            root.next = sibling
            make_next(root.left, root.right)
            make_next(root.right, root.next.left if root.next else None)

        make_next(root, None)

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid or not grid[0]:
            return 0

        R, C = len(grid), len(grid[0])

        def dfs(r, c):
            grid[r][c] = 0
            if r - 1 >= 0 and grid[r - 1][c] == '1':
                dfs(r - 1, c)
            if r + 1 <  R and grid[r + 1][c] == '1':
                dfs(r + 1, c)
            if c - 1 >= 0 and grid[r][c - 1] == '1':
                dfs(r, c - 1)
            if c + 1 <  C and grid[r][c + 1] == '1':
                dfs(r, c + 1)

        ret = 0
        for r in xrange(R):
            for c in xrange(C):
                if grid[r][c] == '1':
                    ret += 1
                    dfs(r, c)

        return ret

    def longestSubstring(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: int
        """
        if k <= 1:
            return s

        def longest(s):
            print s
            if not s:
                return 0

            count = {}
            for c in s:
                if c not in count:
                    count[c] = 0
                count[c] += 1

            if min(count.values()) >= k:
                return len(s)

            start = 0
            ret = 0
            for i in xrange(len(s)):
                if count[s[i]] < k:
                    ret = max(ret, longest(s[start:i]))
                    start = i + 1
            ret = max(ret, longest(s[start:]))
            return ret

        return longest(s)

    def serialize(self, root):
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

    def deserialize(self, data):
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

    def countSmaller(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        n = len(nums)
        ret = [0] * n

        def mergeSortWithCount(nums):
            mid = len(nums) / 2
            if mid <= 0:
                return nums
            leftpart, rightpart = mergeSortWithCount(nums[:mid]), mergeSortWithCount(nums[mid:])
            rightsmaller = 0
            result = []
            leftiter, rightiter = 0, 0
            while leftiter < len(leftpart) and rightiter < len(rightpart):
                if rightpart[rightiter] < leftpart[leftiter]:
                    rightsmaller += 1
                    result.append(rightpart[rightiter])
                    rightiter += 1
                else:
                    ret[leftpart[leftiter][1]] += rightsmaller
                    result.append(leftpart[leftiter])
                    leftiter += 1

            while rightiter < len(rightpart):
                rightsmaller += 1
                result.append(rightpart[rightiter])
                rightiter += 1

            while leftiter < len(leftpart):
                ret[leftpart[leftiter][1]] += rightsmaller
                result.append(leftpart[leftiter])
                leftiter += 1
            return result

        mergeSortWithCount([(val, index) for index, val in enumerate(nums)])
        return ret

    def swimInWater(self, grid):
        """
        :type grid: List[List[int]]
        :rtype: int
        """
        if not grid:
            return 0

        n = len(grid)

        def bfs(limit):
            visited = [[False] * n for i in xrange(n)]
            queue = [(0, 0), ]
            index = 0
            while index < len(queue):
                r, c = queue[index]
                if not visited[r][c]:
                    if grid[r][c] <= limit:
                        if r == n - 1 and c == n - 1:
                            return True
                        visited[r][c] = True
                        if r - 1 >= 0:
                            queue.append((r - 1, c))
                        if r + 1 <  n:
                            queue.append((r + 1, c))
                        if c - 1 >= 0:
                            queue.append((r, c - 1))
                        if c + 1 <  n:
                            queue.append((r, c + 1))
                index += 1
            return False

        left = 2 * n - 2
        right = n * n - 1
        while left < right:
            mid = (left + right) / 2
            if bfs(mid):
                right = mid
            else:
                left = mid + 1
        return left

    def findRedundantConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        n = len(edges)
        if n <= 3:
            return

        disjoint_set = range(n + 1)
        def get_root(i):
            if i != disjoint_set[i]:
                disjoint_set[i] = get_root(disjoint_set[i])
            return disjoint_set[i]

        for u, v in edges:
            if get_root(u) == get_root(v):
                return [min(u, v), max(u, v)]
            else:
                disjoint_set[get_root(v)] = get_root(u)

    def findRedundantDirectedConnection(self, edges):
        """
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        n = len(edges)


        visited = set()
        edge_table = [[] for _ in xrange(n + 1)]
        in_edge = [[] for _ in xrange(n + 1)]

        for u, v in edges:
            edge_table[u].append(v)
            in_edge[v].append(u)

        def check_connectivity_without(exclude_edge):
            groups = [i for i in xrange(n + 1)]

            def get_root(i):
                if groups[i] != i:
                    groups[i] = get_root(groups[i])
                return groups[i]

            for u, v in edges:
                if [u, v] != exclude_edge:
                    groups[get_root(u)] = get_root(v)
            return len(set(get_root(i) for i in xrange(1, n + 1))) == 1

        for v in xrange(1, n + 1):
            if len(in_edge[v]) == 2:
                u1, u2 = in_edge[v]
                if not check_connectivity_without([u1, v]):
                    return [u2, v]
                elif not check_connectivity_without([u2, v]):
                    return [u1, v]
                else:
                    return [u2, v]

        def detect_loop(u, head, prefix):

            if u == head:
                return [prefix, u]

            for v in edge_table[u]:
                visited.add((u, v))
                last_edge = detect_loop(v, head, u)
                if last_edge:
                    return last_edge
            return False

        for u, v in edges:
            if (u, v) not in visited:
                last_edge = detect_loop(v, u, u)
                if last_edge:
                    return last_edge

    def nextGreatestLetter(self, letters, target):
        """
        :type letters: List[str]
        :type target: str
        :rtype: str
        """
        if not (letters[0] <= target < letters[-1]):
            return letters[0]

        left = 0
        right = len(letters)
        while left < right:
            mid = (left + right) / 2
            if letters[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return letters[left]

    def countDigitOne(self, n):
        """
        :type n: int
        :rtype: int
        """
        digit = 0
        ret = 0
        ones = True
        num = 0
        while ones:
            append = 8 * 10 ** digit + 1
            digit += 1
            ones = int(1.0 * (n + append) / 10 ** digit)
            print ones

            ret += ones
        return ret



    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """

        def getAllPalindromes(s):
            n = len(s)
            palindromes = [[] for i in xrange(n)]
            for i, center in enumerate(s):
                l = 0
                while True:
                    # odd length
                    start_i = i - l
                    end_i = i + l
                    if start_i >= 0 and end_i < n and s[start_i] == s[end_i]:
                        palindromes[start_i].append(end_i + 1)
                        # print 'odd ', s[start_i:end_i + 1]
                        l += 1
                    else:
                        break
                l = 0
                while True:
                    # even length
                    start_i = i - l
                    end_i = i + 1 + l
                    if start_i >= 0 and end_i < n and s[start_i] == s[end_i]:
                        palindromes[start_i].append(end_i + 1)
                        # print 'even', s[start_i:end_i + 1]
                        l += 1
                    else:
                        break
            return palindromes

        def partitionString(s, start_i, palindromes):
            if start_i >= len(s):
                return [[]]
            ret = []
            for end_i in palindromes[start_i]:
                palindrome = s[start_i:end_i]
                for subpartition in partitionString(s, end_i, palindromes):
                    ret.append([palindrome, ] + subpartition)
            return ret

        return partitionString(s, 0, getAllPalindromes(s))

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        pre_count = [0] * numCourses
        pos_table = [[] for _ in xrange(numCourses)]
        taken = [False] * numCourses
        for pos, pre in prerequisites:
            pre_count[pos] += 1
            pos_table[pre].append(pos)

        queue = filter(lambda i: pre_count[i] == 0, xrange(numCourses))

        i = 0
        ret = []
        while i < len(queue):
            take_course = queue[i]
            taken[take_course] = True
            ret.append(take_course)
            for course in pos_table[take_course]:
                pre_count[course] -= 1
                if pre_count[course] == 0:
                    queue.append(course)
            i += 1

        return ret if all(taken) else []


    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        ps = []
        stack = []
        for i, c in enumerate(s):
            if c == '(':
                stack.append('(')
            elif c == ')':
                j = len(stack) - 1
                while j >= 0 and stack[j] != '(':
                    j -= 1
                if j < 0:
                    if stack:
                        ps += stack
                        stack = []
                else:
                    tail = ''
                    while len(stack) > j:
                        tail = stack.pop() + tail
                    tail += ')'
                    while stack and len(stack[-1]) > 1:
                        tail = stack.pop() + tail
                    stack.append(tail)
        while stack:
            tail = stack.pop()
            if len(tail) > 1:
                ps.append(tail)
        ret = max(map(len, ps))
        return ret if ret > 1 else 0

    def findMinHeightTrees(self, n, edges):
        """
        :type n: int
        :type edges: List[List[int]]
        :rtype: List[int]
        """
        linkedList = [[] for _ in xrange(n)]
        for u, v in edges:
            linkedList[u].append(v)
            linkedList[v].append(u)

        def getPath(u):
            def _path(u, prev, level):
                next_level = filter(lambda v: v != prev, linkedList[u])
                if not next_level:
                    return [u, ]
                else:
                    ret = []
                    for v in next_level:
                        path = _path(v, u, level + 1)
                        if len(path) > len(ret):
                            ret = path
                    ret.append(u)
                    return ret

            return _path(u, None, 0)

        path = getPath(0)
        startOfLongest = path[0]
        longestPath = getPath(startOfLongest)
        l = len(longestPath)

        if l & 1:
            return [longestPath[l / 2], ]
        else:
            return [longestPath[l / 2], longestPath[l / 2 - 1]]

    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        if not timeSeries:
            return 0

        cumulated = 0
        start = -1
        end = -1
        for t in sorted(timeSeries):
            if t >= end:
                cumulated += end - start
                start = t
                end = t + duration
            else:
                end = t + duration
        cumulated += end - start
        return cumulated

    def canPlaceFlowers(self, flowerbed, n):
        """
        :type flowerbed: List[int]
        :type n: int
        :rtype: bool
        """
        i = 0
        while n and i < len(flowerbed):
            print i, flowerbed[i]
            if flowerbed[i]:
                i += 1
            else:
                print i, flowerbed[i-1]
                if i > 0 and flowerbed[i-1]:
                    i += 1
                else:
                    if i + 1 < len(flowerbed) and not flowerbed[i + 1] or i + 1 == len(flowerbed):
                        flowerbed[i] = 1
                        n -= 1
                    else:
                        i += 1
        print flowerbed
        if n == 0:
            return True
        else:
            return False

    def scheduleCourse(self, courses):
        """
        :type courses: List[List[int]]
        :rtype: int
        """
        courses.sort(key=lambda course: course[1])
        total_time = 0
        maxheap = []
        for time, end in courses:
            if total_time + time <= end:
                total_time += time
                heapq.heappush(maxheap, -time)
            elif maxheap:
                current_longest = -maxheap[0]
                if current_longest > time:
                    total_time = total_time - current_longest + time
                    heapq.heapreplace(maxheap, -time)
        return len(maxheap)

    def buildTree(self, preorder, inorder):
        """
        :type preorder: List[int]
        :type inorder: List[int]
        :rtype: TreeNode
        """

        def build(preorder, inorder):
            if inorder:
                root_val = preorder.pop()
                root_idx = inorder.index(root_val)
                left_inorder = inorder[:root_idx]
                right_inorder = inorder[root_idx + 1:]

                root = TreeNode(root_val)
                root.left = build(preorder, left_inorder)
                root.right = build(preorder, right_inorder)
                return root
            else:
                return None

        return build(preorder[::-1], inorder)

    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """

        trie = {}
        def buildTrie(words):
            for word in words:
                prefix = trie
                for c in word:
                    if c not in prefix:
                        prefix[c] = {}
                    prefix = prefix[c]
                    if prefix.get('#'):
                        break
                prefix['#'] = True
            return trie

        def findRoot(word):
            prefix = trie
            ret = ''
            for c in word:
                if c not in prefix:
                    return word
                else:
                    ret += c
                    prefix = prefix[c]
                    if prefix.get('#'):
                        return ret
            return word

        buildTrie(dict)
        words = sentence.split(' ')
        return ' '.join(map(findRoot, words))

    def wordBreak(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: bool
        """

        possibleLength = set(map(len, wordDict))
        wordSet = set(wordDict)
        canBreak = [False] * (len(s))

        for i in xrange(len(s)):
            for l in possibleLength:
                if i - l + 1 == 0:
                    canBreak[i] = s[i - l + 1:i + 1] in wordSet
                elif i - l + 1 > 0:
                    canBreak[i] = s[i - l + 1:i + 1] in wordSet and canBreak[i - l]
                else:
                    continue
                if canBreak[i]:
                    break
        return canBreak[-1]

    def wordBreak2(self, s, wordDict):
        breaks = [None] * (len(s) + 1)
        wordSet = set(wordDict)
        def breakWord(start):
            if start == len(s):
                return ['']
            if breaks[start] is None:
                breaks[start] = [
                    s[start:j] + (' ' + tail if tail else '')
                    for j in xrange(start + 1, len(s)+1)
                    if s[start:j] in wordSet
                    for tail in breakWord(j)
                ]
            return breaks[start]
        return breakWord(0)

    def wordBreak3(self, s, wordDict):
        """
        :type s: str
        :type wordDict: List[str]
        :rtype: List[str]
        """

        possibleLength = set(map(len, wordDict))
        wordSet = set(wordDict)
        canBreak = [[] for c in s]
        memo = dict()

        for i in xrange(len(s)):
            for l in possibleLength:
                word = s[i - l + 1:i + 1]
                if i - l + 1 == 0:
                    if (i - l + 1, i + 1) not in memo:
                        memo[(i - l + 1, i + 1)] = word in wordSet
                    if memo[(i - l + 1, i + 1)]:
                        canBreak[i].append(word)
                elif i - l + 1 > 0:
                    if (i - l + 1, i + 1) not in memo:
                        memo[(i - l + 1, i + 1)] = word in wordSet
                    if memo[(i - l + 1, i + 1)]:
                        for brk in canBreak[i - l]:
                            canBreak[i].append(brk + ' ' + word)
                else:
                    continue

        return canBreak[-1]

    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """

        def reverseList(head):
            def _recusive(head):
                if head.next is None:
                    return head, head
                n, next = head, head.next
                n.next = None
                h, t = _recusive(next)
                t.next = n
                return h, n
            if head is None:
                return head
            return _recusive(head)[0]

        if not head or not head.next:
            return True

        h1 = h2 = head
        while h2:
            h2 = h2.next
            if h2:
                h2 = h2.next
                h1 = h1.next
            else:
                break

        tail_of_first_part = head
        while tail_of_first_part.next != h1:
            tail_of_first_part = tail_of_first_part.next
        tail_of_first_part.next = None

        H1 = ListNode(0)
        H1.next = head
        H2 = ListNode(0)
        H2.next = reverseList(h1)
        h1, h2 = H1.next, H2.next
        while h1 and h2:
            if h1.val != h2.val:
                return False
            else:
                h1 = h1.next
                h2 = h2.next

        # recover
        tail_of_first_part.next = reverseList(H2.next)

        return True

    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        import heapq

        n = len(nums)
        if not n:
            return []
        if k == 1:
            return nums

        ret = []
        window = [(-nums[i], i) for i in xrange(k)]
        heapq.heapify(window)
        ret.append(-window[0][0])

        for i in xrange(k, n):
            heapq.heappush(window, (-nums[i], i))
            left = i - k
            topv, topi = window[0]
            while topi <= left:
                heapq.heappop(window)
                topv, topi = window[0]
            ret.append(-topv)
        return ret

    def maxSlidingWindow2(self, nums, k):
        import collections

        if not nums:
            return []

        max_candidate_indexes_within_window = collections.deque()
        ret = []
        for i, v in enumerate(nums):
            while max_candidate_indexes_within_window and nums[max_candidate_indexes_within_window[-1]] < v:
                max_candidate_indexes_within_window.pop()
            max_candidate_indexes_within_window.append(i)
            while max_candidate_indexes_within_window and max_candidate_indexes_within_window[0] <= i - k:
                max_candidate_indexes_within_window.popleft()
            if i >= k - 1:
                ret.append(nums[max_candidate_indexes_within_window[0]])

        return ret

    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

        if not s or not t:
            return ''

        requires = {}
        for c in t:
            if c not in requires:
                requires[c] = 0
            requires[c] += 1

        left = 0
        right = 0
        key_indexes = []
        key_i = 0
        ret = None
        while right < len(s) or max(requires.values()) <= 0:
            if max(requires.values()) > 0:
                # expanding window
                if s[right] in requires:
                    key_indexes.append(right)
                    requires[s[right]] -= 1
                right += 1
            else:
                # shrinking window
                while max(requires.values()) <= 0:
                    left = key_indexes[key_i]
                    key_i += 1
                    if ret is None or right - left < len(ret):
                        ret = s[left:right]
                    requires[s[left]] += 1
        return ret or ''

    def smallestGoodBase(self, n):
        import math
        n = int(n)

        for k in xrange(int(math.log(n, 2)), 1, -1):
            a = int(n ** k ** -1)  # k√n
            if (1 - a ** (k + 1)) / (1 - a) == n:  # [a^0 + a^1 + ... + a^k] == n
                return str(a)

        return str(n - 1)

    def findRadius(self, houses, heaters):
        """
        :type houses: List[int]
        :type heaters: List[int]
        :rtype: int
        """
        heaters.sort()
        ranges = [0, ]
        for i in xrange(len(heaters) - 1):
            ranges.append((heaters[i + 1] + heaters[i]) / 2.0)
        ranges.append(10 ** 9)
        def findClosest(pos):
            i = 0
            while not (ranges[i] <= pos < ranges[i + 1]):
                i += 1
            return heaters[i]
        return max(abs(findClosest(pos) - pos) for pos in houses)

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """
        numstack = []
        opstack = []

        num = 0
        for i in xrange(len(s)):
            c = s[i]
            if c.isdigit():
                num = 10 * num + ord(c) - ord('0')
            if c in ['+', '-', '*', '/'] or i == len(s) - 1:
                numstack.append(num)
                num = 0
                if opstack and opstack[-1] in ['*', '/']:
                    op = opstack.pop()
                    num2 = numstack.pop()
                    num1 = numstack.pop()
                    if op == '/':
                        result = num1 / num2
                    else:
                        result = num1 * num2
                    numstack.append(result)
                if i != len(s) - 1:
                    opstack.append(c)

        num1 = numstack[0]
        for i in xrange(len(opstack)):
            op = opstack[i]
            num2 = numstack[i + 1]
            if op == '+':
                num1 += num2
            else:
                num1 -= num2

        return num1

    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        n = len(gas)
        if n <= 0:
            return -1
        if n == 1:
            return 0 if gas[0] >= cost[0] else -1

        n = len(gas)
        if n <= 0:
            return -1
        if n == 1:
            return 0 if gas[0] >= cost[0] else -1

        remainings = [(cost[i] - gas[i], i) for i in xrange(n)]
        remainings.sort()

        impossibles = set()

        for remaining, start in remainings:
            if start in impossibles:
                continue
            if remaining > 0:
                return -1
            i = start
            passby = set()
            g = gas[i] - cost[i]
            i = (i + 1) % n
            while g >= 0 and i != start:
                passby.add(i)
                g += (gas[i] - cost[i])
                i = (i + 1) % n
            if g >= 0 and i == start:
                return start
            else:
                impossibles |= passby

        return -1

    def calculate(self, s):
        """
        :type s: str
        :rtype: int
        """

        stack = []
        left = None
        op = None
        num = None
        i = 0
        while i < len(s):
            if s[i].isdigit():
                num = 0
                while i < len(s) and s[i].isdigit():
                    num = 10 * num + ord(s[i]) - ord('0')
                    i += 1
                if left is None:
                    left = num
                else:
                    left = left + (num if op == '+' else -num)
                    op = None
            elif s[i] in '+-':
                op = s[i]
                i += 1
            elif s[i] == '(':
                if left is not None:
                    stack.append((left, op))
                    left = op = None
                i += 1
            elif s[i] == ')':
                num = left
                if stack:
                    left, op = stack.pop()
                else:
                    left, op = None, None
                if left is None:
                    left = num
                else:
                    left = left + (num if op == '+' else -num)
                    op = None
                i += 1
        return left


    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        def pathTo(root, leave):
            if not root:
                return None
            else:
                if root.val == leave:
                    return [leave, ]
                else:
                    l = pathTo(root.left, leave)
                    r = pathTo(root.right, leave)
                    if l:
                        l.append(root.val)
                        return l
                    elif r:
                        r.append(root.val)
                        return r
                    return None
        path_p = pathTo(root, p)[::-1]
        path_q = pathTo(root, q)[::-1]
        i = 0

        while i < len(path_q) and i < len(path_p) and path_p[i] == path_q[i]:
            i += 1
        return path_p[i-1]

    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x == 0:
            return 0
        elif 0 < x <= 1:
            return 1
        elif 1 < x <= 4:
            return 2

        l, r = 0, x / 2
        while l < r:
            m = (l + r) / 2
            mm = m * m
            if mm == x:
                return m
            elif mm > x:
                r = m
            else:
                l = m + 1
        return l

    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        ops = {
            '+': int.__add__,
            '-': int.__sub__,
            '*': int.__mul__,
            '/': lambda a, b: int(1. * a / b),
        }
        stack = []

        for token in tokens:
            if token in ops:
                n2 = stack.pop()
                n1 = stack.pop()
                stack.append(ops[token](n1, n2))
                print n1, token, n2
                print stack
            else:
                stack.append(int(token))
                print stack
        print stack


    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        n = len(heights)
        lowerLeft = [-1] * n
        lowerRight = [n] * n

        for i in xrange(1, n):
            j = i - 1
            while j >= 0 and heights[j] >= heights[i]:
                j = lowerLeft[j]
            lowerLeft[i] = j

        for i in xrange(n-2, -1, -1):
            j = i + 1
            while j < n and heights[j] >= heights[i]:
                j = lowerRight[j]
            lowerRight[i] = j

        ret = 0
        for i in xrange(n):
            ret = max(ret, (lowerRight[i] - lowerLeft[i] - 1) * heights[i])
        return ret

    def largestRectangleArea2(self, height):
        height.append(0)
        # a guard to guarantee all height in stack pop out
        stack = [-1]
        # a guard to guarantee stack never empty
        ans = 0
        for i in xrange(len(height)):
            # stack would be ascending
            while height[i] < height[stack[-1]]:
                # previous higher is never use again, as current one is lower!
                h = height[stack.pop()]
                w = i - stack[-1] - 1
                print h, w
                ans = max(ans, h * w)
            print '--'
            stack.append(i)

        return ans

    def isPalindrome2(self, s):
        """
        :type s: str
        :rtype: bool
        """
        l, r = 0, len(s) - 1
        while l < r:
            if not s[l].isalnum():
                l += 1
                continue
            if not s[r].isalnum():
                r -= 1
                continue
            if s[l].lower() != s[r].lower():
                return False
            else:
                l += 1
                r -= 1
        return True

    def coinChange(self, coins, amount):
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """

        if amount == 0:
            return 0

        count = [None] * (1 + amount)
        for coin in coins:
            if coin <= amount:
                count[coin] = 1

        for i in xrange(1 + amount):
            if count[i] is not None:
                for coin in coins:
                    newi = i + coin
                    if newi <= amount:
                        newcount = count[i] + 1
                        if count[newi] is None or count[newi] > newcount:
                            count[newi] = newcount
        return count[-1] if count[-1] else -1

    def getSkyline(self, buildings):
        import bisect
        index = [[-1, 0], ]

        for L, R, H in buildings:
            end = bisect.bisect_left(index, [R, ])
            original_end = index[end-1][1]
            start = bisect.bisect_left(index, [L, ])
            if start >= len(index) or index[start][0] != L:
                index.insert(start, [L, index[start-1][1]])
                end += 1
            if end >= len(index) or index[end][0] != R:
                index.insert(end, [R, original_end])
            for i in xrange(start, end):
                if i == start:
                    index[i][1] = max(H, index[i][1])
                elif index[i][1] < H:
                    index[i][1] = H

        ret = []
        for item in index[1:]:
            if not ret or item[1] != ret[-1][1]:
                ret.append(item)
        return ret

    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        R = len(board)
        if not R:
            return word == ''
        C = len(board[0])
        if not C:
            return word == ''

        def dfs(r, c, word):
            if not word:
                return True
            if not (0 <= r < R) or not (0 <= c < C):
                return False
            if board[r][c] != word[0]:
                return False
            board[r][c] = None
            remaining = word[1:]
            if dfs(r - 1, c, remaining):
                return True
            if dfs(r + 1, c, remaining):
                return True
            if dfs(r, c - 1, remaining):
                return True
            if dfs(r, c + 1, remaining):
                return True
            board[r][c] = word[0]
            return False

        for r in xrange(R):
            for c in xrange(C):
                if dfs(r, c, word):
                    return True
        return False

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        if not matrix:
            return []

        def nextdirection(curdir):
            return {
                +1: +2,
                +2: -1,
                -1: -2,
                -2: +1,
            }[curdir]

        def nextstep(curdir):
            return {
                +1: (lambda (r, c): (r, c + 1)),
                +2: (lambda (r, c): (r + 1, c)),
                -1: (lambda (r, c): (r, c - 1)),
                -2: (lambda (r, c): (r - 1, c)),
            }[curdir]

        maxstep = [0, len(matrix[0]), len(matrix)]
        direction = +1
        point = (0, -1)
        ret = []
        step = 1
        while step:
            absdir = abs(direction)
            step = maxstep[absdir]
            maxstep[2 if absdir == 1 else 1] -= 1
            nextfunc = nextstep(direction)
            for _ in xrange(step):
                point = r, c = nextfunc(point)
                ret.append(matrix[r][c])

            direction = nextdirection(direction)
        return ret


    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def maxPathFromRoot(root):
            if not root:
                return 0
            if not hasattr(root, 'mfr'):
                l = maxPathFromRoot(root.left)
                r = maxPathFromRoot(root.right)
                ret = max(l, r, 0) + root.val
                setattr(root, 'mfr', ret)

            return root.mfr

        def maxPathAnyway(root):

            if root.left and root.right:
                return max(
                    maxPathAnyway(root.left),
                    maxPathAnyway(root.right),
                    maxPathFromRoot(root.left) + maxPathFromRoot(root.right) + root.val
                )
            elif not root.left and root.right:
                return max(
                    maxPathAnyway(root.right),
                    maxPathFromRoot(root.right) + root.val,
                )
            elif root.left and not root.right:
                return max(
                    maxPathAnyway(root.left),
                    maxPathFromRoot(root.left) + root.val,
                )
            else:
                return root.val

        return maxPathAnyway(root)

    def largestNumber(self, nums):

        def compare(n1, n2):
            num1 = int(str(n1) + str(n2))
            num2 = int(str(n2) + str(n1))
            if num1 > num2:
                return 1
            elif num1 < num2:
                return -1
            else:
                return 0

        nums.sort(cmp=compare, reverse=True)

        return ''.join(map(str, nums))

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        memo = dict()

        def decode(s):
            if s in memo:
                return memo[s]
            if not s:
                return 1
            if s == 1:
                return 1 if s != '0' else 0
            if s[0] == '0':
                return 0
            else:
                ret = decode(s[1:])
                if len(s) >= 2:
                    if 10 <= int(s[:2]) <= 26:
                        ret += decode(s[2:])
                memo[s] = ret
                return ret
        return decode(s)


    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        allChar = 'qazwsxedcrfvtgbyhnujmkiolp'
        lenWord = len(beginWord)
        wordSet = set(wordList)

        memo = {}
        def allDiffOne(word):
            if word not in memo:
                ret = set()
                for i in xrange(lenWord):
                    for c in allChar:
                        if c != word[i]:
                            ret.add(word[:i] + c + word[i+1:])
                memo[word] = ret
            return memo[word]

        if endWord not in wordSet:
            return []

        forward = [[beginWord, ]]
        forwardSet = {beginWord, }
        backward = [[endWord, ]]
        backwardSet = {endWord, }

        while forward:
            if forwardSet & backwardSet:
                break

            if len(forward) > len(backward):
                forward, forwardSet, backward, backwardSet = backward, backwardSet, forward, forwardSet

            wordSet -= forwardSet
            newForward = []
            newForwardSet = set()
            for p in forward:
                for word in allDiffOne(p[-1]) & wordSet:
                    newForwardSet.add(word)
                    newForward.append(p + [word, ])

            forward, forwardSet = newForward, newForwardSet

        if not forward or not backward:
            return []
        if forward[0][0] != beginWord:
            forward, backward = backward, forward

        ret = []
        for p2 in backward:
            tail = p2[:-1][::-1]
            for p1 in forward:
                if p1[-1] == p2[-1]:
                    ret.append(p1 + tail)
        return ret

    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        nums.sort()

        k = len(nums) - 1
        lastLargest = None
        ret = []
        while k >= 0 and nums[k] >= 0:
            if lastLargest == nums[k]:
                k -= 1
                continue
            else:
                lastLargest = nums[k]
            i, j = 0, k - 1
            target = - nums[k]
            while i < j and nums[i] < 0:
                if nums[i] + nums[j] == target:
                    new = [nums[i], nums[j], nums[k]]
                    if not ret or ret[-1] != new:
                        ret.append(new)
                    i += 1
                elif nums[i] + nums[j] > target:
                    j -= 1
                else:
                    i += 1
        return ret

    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        n = len(points)
        if n <= 1:
            return n

        ret = 0
        for i in xrange(n):
            pi = points[i]
            count = {'inf': 1}
            same = 0
            count['inf'] = 1
            for j in xrange(i+1, n):
                pj = points[j]
                if pi.x == pj.x:
                    if pi.y == pj.y:
                        same += 1
                        continue
                    else:
                        count['inf'] += 1
                else:
                    k = 1.0 * (pj.y - pi.y) / (pj.x - pi.x)
                    if k not in count: count[k] = 1
                    count[k] += 1
            kmax = max(count.values()) if count else 0
            print count
            ret = max(ret, kmax + same)
        return ret

    def maxPoints2(self, points):
        l = len(points)
        m = 0
        for i in range(l):
            dic = {'i': 1}
            same = 0
            for j in range(i+1, l):
                tx, ty = points[j].x, points[j].y
                if tx == points[i].x and ty == points[i].y:
                    same += 1
                    continue
                if points[i].x == tx: slope = 'i'
                else:slope = (points[i].y-ty) * 1.0 /(points[i].x-tx)
                if slope not in dic: dic[slope] = 1
                dic[slope] += 1
            print dic
            m = max(m, max(dic.values()) + same)

        return m

    def isMatch(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        patterns = []
        stars = set()
        for i in xrange(len(p)):
            if p[i] == '*':
                stars.add(len(patterns) - 1)
            else:
                patterns.append(p[i])

        i, j = 0, 0
        while i < len(s) and j < len(patterns):
            if s[i] == patterns[j]:
                i += 1
                if j not in stars:
                    j += 1
            else:
                pass

            return False

        if i < len(s) or j < len(patterns):
            return False

        return True

    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        str = str.strip()
        if not str:
            return 0
        sign = +1
        if str[0] == '+' or str[0] == '-':
            sign = int(str[0] + '1')
            str = str[1:]
        absnum = 0
        for c in str:
            if c.isdigit():
                absnum = 10 * absnum + ord(c) - ord('0')
            else:
                break
        num = sign * absnum
        if num < - 2 ** 31:
            return - 2 ** 31
        elif num > 2 ** 31 - 1:
            return 2 ** 31 - 1
        return num

    def wiggleSort(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n <= 1:
            return
        nums.sort()

        result = [0] * n
        half = (n - 1) / 2
        i = half
        x = 0
        while i >= 0:
            result[x] = nums[i]
            x += 2
            i -= 1

        j = half + 1
        x = 1
        while j < n:
            result[x] = nums[j]
            x += 2
            j += 1

        nums[:] = result

        print nums

if __name__ == '__main__':
    print Solution().largestRectangleArea2([2,1,5,6,2,3])
