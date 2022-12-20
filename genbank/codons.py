

class Next:
	def __init__(self, n=None):
		self.next = n
	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__, self.next)

class Last:
	def __init__(self, n=None):
		self.last = n
	def __repr__(self):
		return "%s(%s)" % (self.__class__.__name__, self.last)

class Codons(dict):
	def abs(self, n, item):
		distance = abs(item - n) if item else float("+Inf")
		return distance

	def nearest(self, n, lst):
		nearest = lst[0] if self.abs(n, lst[0]) < self.abs(n, lst[1]) else lst[1]
		return nearest

	def nearest_start(self, n, strand):
		n = n - 1
		lst = [ self.last_start[n].last , self.next_start[n].next ] if strand=='+' else [ self.last_start_[n].last , self.next_start_[n].next ]
		return self.nearest(n, lst) + 1

	def nearest_stop(self, n, strand):
		n = n - 1
		lst = [ self.last_stop[n].last , self.next_stop[n].next ] if strand=='+' else [ self.last_stop_[n].last , self.next_stop_[n].next ]
		return self.nearest(n, lst) + 1

	def last_start(self, n, strand):
		n = n - 1
		return self.last_start[n]

	def __init__(self, dna):
		pass


	def ininti(self, codons):
		self.last_start = [None] * len(dna)
		self.last_stop  = [None] * len(dna)
		self.next_start = [None] * len(dna)
		self.next_stop  = [None] * len(dna)
		self.last_start_ = [None] * len(dna)
		self.last_stop_  = [None] * len(dna)
		self.next_start_ = [None] * len(dna)
		self.next_stop_  = [None] * len(dna)
	
		lstarts = {0: Last(-1), 1: Last(-1), 2: Last(-1) }
		lstops  = {0: Last(-1), 1: Last(-1), 2: Last(-1) }
		nstarts = {0: Next(), 1: Next(), 2: Next() }
		nstops  = {0: Next(), 1: Next(), 2: Next() }
		lstarts_ = {0: Last(-1), 1: Last(-1), 2: Last(-1) }
		lstops_  = {0: Last(-1), 1: Last(-1), 2: Last(-1) }
		nstarts_ = {0: Next(), 1: Next(), 2: Next() }
		nstops_  = {0: Next(), 1: Next(), 2: Next() }

		i = 0
		for i,_ in enumerate(dna):
			codon = dna[i:i+3]
			frame = i % 3

			self.next_start[i] = nstarts[frame]
			# STARTS
			if codon in ['atg','gtg','ttg']:
				# last
				lstarts[frame] = Last(i)
				# next
				nstarts[frame].next = i
				nstarts[frame] = Next(i)
			self.last_start[i] = lstarts[frame]
			#---------------------------------
			self.next_start_[i] = nstarts_[frame]
			if codon in ['cat','cac','caa']:
				# last
				lstarts_[frame] = Last(i)
				# next
				nstarts_[frame].next = i
				nstarts_[frame] = Next(i)
			self.last_start_[i] = lstarts_[frame]

			# STOPS
			self.next_stop[i] = nstops[frame]
			if codon in ['tag','tga','taa']:
				# last
				lstarts[frame] = Last(i)
				lstops[frame] = Last(i)
				# next
				nstops[frame].next = i
				nstops[frame] = Next(i)
			self.last_stop[i] = lstops[frame]
			#---------------------------------
			self.next_stop_[i] = nstops_[frame]
			if codon in ['cta','tca','tta']:
				# last
				lstarts_[frame] = Last(i)
				lstops_[frame] = Last(i)
				# next
				nstops_[frame].next = i
				nstops_[frame] = Next(i)
			self.last_stop_[i] = lstops_[frame]
		# for orfs that run off the ends
		for frame in [0,1,2]:
			print(i-frame)
			self.next_stop[i-frame].next = i + 3
			self.next_start[i-frame].next = i + 3
			self.next_stop_[i-frame].next = i + 3
			self.next_start_[i-frame].next = i + 3
		#for i, obj in enumerate(self.next_stop):
		#	print(i, obj)



