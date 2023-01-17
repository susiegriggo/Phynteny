from itertools import zip_longest, chain
import textwrap
import copy

def rev_comp(dna):
	a = 'acgtrykmbvdh'
	b = 'tgcayrmkvbhd'
	tab = str.maketrans(a,b)
	return dna.translate(tab)[::-1]

def grouper(iterable, n, fillvalue=None):
	args = [iter(iterable)] * n
	return zip_longest(*args, fillvalue=fillvalue)

def nint(s):
	return int(s.replace('<','').replace('>',''))


class Feature():
	def __init__(self, type_, strand, pairs, locus, tags=None):
		#super().__init__(locus.locus, locus.dna)
		self.type = type_
		self.strand = strand
		# tuplize the pairs
		self.pairs = tuple([tuple(pair) for pair in pairs])
		self.locus = locus
		self.tags = tags if tags else dict()
		self.dna = ''
		self.partial = False
	
	def length(self):
		return len(self.seq())

	def seq(self):
		seq = ''
		for n in self.base_locations():
			seq += self.locus.seq(n,n, self.strand) #TODO this doesn't work because selecting from n to n
			print(self.locus.seq)
		if self.strand > 0:
			return seq
		else:
			return seq[::-1]

	def fna(self):
		return self.header() + self.seq() + "\n"

	def faa(self):
		return self.header() + self.translation() + "\n"
	
	def header(self):
		header = ">" + self.locus.name() + "_CDS_[" + self.locations() + "]"
		for tag in self.tags:
			header += " [" + tag + "=" + self.tags[tag] +"]"
		return header + "\n"

	def frame(self, end):
		if self.type != 'CDS':
			return 0
		elif end == 'right':
			return (self.right()%3+1) * self.strand
		elif end == 'left':
			return (self.left()%3+1) * self.strand

	def hypothetical(self):
		function = self.tags['product'] if 'product' in self.tags else ''
		if 'hypot'  in function or \
		   'etical' in function or \
		   'unchar' in function or \
		   ('orf' in function and 'orfb' not in function):
			return True
		else:
			return False

	def partial(self):
		partial_left  = any(['<' in item for pair in self.pairs for item in pair])
		partial_right = any(['>' in item for pair in self.pairs for item in pair])
		if partial_left and partial_right:
			# this really shouldnt even happen, maybe raise an error?
			return 'both'
		elif partial_left:
			return 'left'
		elif partial_right:
			return 'right'
		else:
			return None

	def is_type(self, _type):
		if self.type == _type:
			return True
		else:
			return False

	def left(self):
		# convert genbank 1-based indexing to standard 0-based
		return nint(self.pairs[0][0]) - 1
	
	def right(self):
		# convert genbank 1-based indexing to standard 0-based
		return nint(self.pairs[-1][-1]) - 3

	def is_joined(self):
		if len(self.pairs) > 1:
			return True
		return False

	def __iter__(self):
		for left,*right in self.pairs:
			if right:
				right = right[0]
			else:
				right = left
			yield nint(left) , nint(right)

	def __str__(self):
		"""Compute the string representation of the feature."""
		return "%s\t%s\t%s\t%s" % (
				repr(self.locus.name()),
				repr(self.type),
				repr(self.pairs),
				repr(self.tags))

	def __repr__(self):
		"""Compute the string representation of the feature."""
		return "%s(%s, %s, %s, %s)" % (
				self.__class__.__name__,
				repr(self.locus.name()),
				repr(self.type),
				repr(self.pairs),
				repr(self.tags))
	def __hash__(self):
		return hash(self.pairs)
	#def __eq__(self, other):
	#	return self.pairs == other.pairs()

	def __lt__(self, other):
		if self.left() == other.left():
			return self.right() < other.right()
		else:
			return self.left() < other.left()

	def locations(self):
		pairs = []
		#for left, *right in self.pairs:
		for pair in self.pairs:
			pairs.append("..".join(pair))
		location = ','.join(pairs)
		if len(pairs) > 1:
			location = 'join(' + location + ')'
		if self.strand < 0:
			location = 'complement(' + location + ')'
		return location


	def base_locations(self, full=False):
		if full and self.partial == 'left': 
			for i in range(-((3 - len(self.dna) % 3) % 3), 0, 1):
				yield i+1
		for left,right in self:
			#left,right = map(int, [ item.replace('<','').replace('>','') for item in self.pair ] )
			for i in range(left,right+1):
				if i <= self.locus.length():
					yield i

	def codon_locations(self):
		assert self.type == 'CDS'
		for triplet in grouper(self.base_locations(full=True), 3):
			if triplet[0] >= 1:
				yield triplet

	def codons(self):
		assert self.type == 'CDS'
		for locations in self.codon_locations():
			if self.strand > 0:
				yield ''.join([self.locus.dna[loc-1] if loc else '' for loc in locations])
			else:
				yield rev_comp(''.join([self.locus.dna[loc-1] if loc else '' for loc in locations]))

	def split(self):
		a = copy.copy(self)
		b = copy.copy(self)
		return a,b

	def translation(self):
		aa = []
		codon = ''
		first = 0 if not self.partial else len(self.dna) % 3
		dna = self.seq()
		for i in range(first, self.length(), 3):
			codon = dna[ i : i+3 ]
			aa.append(self.locus.translate.codon(codon))
		#if self.strand < 0:
		#	aa = aa[::-1]
		# keeping the stop codon character adds 'information' as does which of
		# the stop codons it is. It is the better way to write the fasta
		#if aa[-1] in '#*+':
		#	aa.pop()
		# keeping the first amino acid also adds 'information' to the fasta
		#aa[0] = 'M'
		return "".join(aa)

	def write(self, outfile):
		outfile.write('     ')
		outfile.write( self.type.ljust(16) )
		if self.strand < 0:
			outfile.write('complement(')
		# the pairs
		if len(self.pairs) > 1:
			outfile.write('join(')
		pairs = []
		#for left, *right in self.pairs:
		for pair in self.pairs:
			#pair = left + '..' + str(right[0]) if right else str(left)
			pairs.append("..".join(pair))
		outfile.write(','.join(pairs))
		if len(self.pairs) > 1:
			outfile.write(')')
		# the pairs
		if not self.strand > 0:
			outfile.write(')')
		outfile.write('\n')
		for tag,values in self.tags.items():
			for value in values:
				if value is not None:
					for line in textwrap.wrap( '/' + str(tag) + '=' + str(value) , 58):
						outfile.write('                     ')
						outfile.write(line)
						outfile.write('\n')
				else:
					outfile.write('                     ')
					outfile.write('/' + str(tag))
					outfile.write('\n')

