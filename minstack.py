class MinStack(object):
	def __init__(self):
		"""
		initialize your data structure here.
		"""
		self.stack = []
		self.min_index = []

	def push(self, x):
		"""
		:type x: int
		:rtype: void
		"""
		if not self.stack or x < self.getMin():
			self.min_index.append(len(self.stack))
		self.stack.append(x)

	def pop(self):
		"""
		:rtype: void
		"""
		if self.min_index[-1] == len(self.stack) - 1:
			self.min_index.pop()
		self.stack.pop()

	def top(self):
		"""
		:rtype: int
		"""
		return self.stack[-1]

	def getMin(self):
		"""
		:rtype: int
		"""
		return self.stack[self.min_index[-1]]


s = MinStack()
s.push(-2)
s.push(0)
s.push(-3)
print s.stack
print s.min_index