from linked_list import make_list, ListNode
from tree import TreeNode

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

print Solution().decodeString("2[y]pq4[2[jk]e1[f]]")