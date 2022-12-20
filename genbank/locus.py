import re
import sys
import textwrap
from collections.abc import Sequence

from genbank.codons import Last
from genbank.codons import Next
from genbank.codons import Codons
from genbank.feature import Feature
from genbank.translate import Translate

def rev_comp(dna):
	a = 'acgtrykmbvdh'
	b = 'tgcayrmkvbhd'
	tab = str.maketrans(a,b)
	return dna.translate(tab)[::-1]

def nint(s):
	return int(s.replace('<','').replace('>',''))

def rmap(func, items):
	out = list()
	for item in items:
		if isinstance(item, Sequence) and not isinstance(item, str):
			out.append(type(item)(rmap(func,item)))
		else:
			out.append(func(item))
	return type(items)(out)

def recursive_map(func, items):
    return (recursive_map(func, x) if isinstance(x, tuple) else func(x) for x in items)


class Seq(str):
	# this is just to capture negative string indices as zero
	def __getitem__(self, key):
		if isinstance(key, slice) and key.start < 0:
			key = slice(0, key.stop, key.step)
		elif isinstance(key, int) and key >= len(self):
			return ''
		return super().__getitem__(key)

class Locus(dict):
	def __init__(self, name='', dna=''):
		if not hasattr(self, 'feature'):
			self.feature = Feature
		#self.name = name
		self.dna = dna.lower()
		#self.codons = dict()
		self.translate = Translate()
		self.strand = 1
		self.groups = dict()
		self.groups['LOCUS'] = [name.replace(' ','')]
		#self.groups['FEATURES'] = ['']
		#self.groups['ORIGIN'] = ['']

	def __init_subclass__(cls, feature=Feature, **kwargs):
		'''this method allows for a Feature class to be modified through inheritance in other code '''
		super().__init_subclass__(**kwargs)
		cls.feature = feature

	def name(self):
		return self.groups['LOCUS'][0].split()[0]
	def molecule(self):
		if len(locus) > 2:
			return locus[3]
		else:
			return 'DNA'
	def locus(self):
		cols = self.groups['LOCUS'][0].split(' ')
		# I eventually need to properly format the locus line
		locus =  self.name().ljust(9)
		locus += str(len(self.dna)).rjust(19)
		locus += ' bp '
		if 'bp' in cols:
			locus += ' '.join(cols[cols.index('bp')+1:])
		else:
			locus += '\n'
		return locus

		

	def fasta(self):
		return ">" + self.name() + "\n" + self.seq() + "\n"

	def seq(self, left=0, right=None, strand=None):
		# this should always refer to zero based indexing
		if strand is None:
			strand = self.strand
		if left < 0:
			left = 0
		if right is None:
			right = self.length() - 1
		if strand > 0:
			return Seq(         self.dna[left : right])
		else:
			return Seq(rev_comp(self.dna[left : right]))

	def length(self):
		return len(self.dna)

	def gc_content(self, seq=None):
		if seq is not None:
			#a = seq.count('a') + seq.count('A')
			c = seq.count('c') + seq.count('C')
			g = seq.count('g') + seq.count('G')
			#t = seq.count('t') + seq.count('T')
			return (c+g) / len(seq) #(a+c+g+t)
		elif not hasattr(self, "gc"):
			#a = self.dna.count('a') + self.dna.count('A')
			c = self.dna.count('c') + self.dna.count('C')
			g = self.dna.count('g') + self.dna.count('G')
			#t = self.dna.count('t') + self.dna.count('T')
			self.gc = (c+g) / len(self.dna) # (a+c+g+t)
		return self.gc

	def pcodon(self, codon):
		codon = codon.lower()
		seq = self.dna + rev_comp(self.dna)
		p = dict()
		p['a'] = seq.count('a') / len(seq)
		p['c'] = seq.count('c') / len(seq)
		p['g'] = seq.count('g') / len(seq)
		p['t'] = seq.count('t') / len(seq)
		return p[codon[0]] * p[codon[1]] * p[codon[2]]

	def rbs(self):
		for feature in self:
			if feature.type == 'CDS':
				if feature.strand > 0:
					start = feature.left()+3
					feature.tags['rbs'] = self.seq(start-30,start)
				else:
					start = feature.right()
					feature.tags['rbs'] = rev_comp(self.seq(start,start+30))
	
	def features(self, include=None, exclude=None):
		for feature in self:
			if not include or feature.type in include:
				yield feature

	def add_feature(self, key, strand, pairs, tags=dict()):
		"""Add a feature to the factory."""
		#feature = self.feature
		feature = self.feature(key, strand, pairs, self)
		if feature not in self:
			self[feature] = len(self)
		return feature

	def read_feature(self, line):
		"""Add a feature to the factory."""
		key = line.split()[0]
		val = line.split()[1]
		#partial  = 'left' if '<' in line else ('right' if '>' in line else False)
		strand = -1 if 'complement' in line else 1
		# this is for weird malformed features
		if ',1)' in line:
			line = line.replace( ",1)" , ",1..1)" )
		#pairs = [pair.split('..') for pair in re.findall(r"<*\d+\.\.>*\d+", line)]
		#pairs = [map(int, pair.split('..')) for pair in re.findall(r"<?\d+\.{0,2}>?\d+", line.replace('<','').replace('>','') )]
		#pairs = [ pair.split('..') for pair in re.findall(r"<?\d+\.{0,2}>?[-0-9]*", line) ]
		pairs = [ pair.split('..') for pair in re.findall(r"<?\d+\.{0,2}>?\d*", val) ]
		# tuplize the pairs
		pairs = tuple([tuple(pair) for pair in pairs])
		feature = self.add_feature(key, strand, pairs)
		return feature

	def gene_coverage(self):
		''' This calculates the protein coding gene coverage, which should be around 1 '''
		cbases = tbases = 0	
		for locus in self.values():
			dna = [False] * len(self.dna)
			seen = dict()
			for feature in self.features(include=['CDS','tRNA']):
				#for locations in feature.codon_locations():
				for location in feature.base_locations():
					if location:
						dna[location-1] = True
			cbases += sum(dna)
			tbases += len(dna)
		return cbases / tbases

	def write(self, outfile=sys.stdout):
		for group,values in self.groups.items():
			for value in values:
				if group == 'LOCUS':
					outfile.write('LOCUS       ')
					outfile.write(self.locus())
				elif group == 'FEATURES' or 'FEATURES' not in self.groups:
					outfile.write('FEATURES             Location/Qualifiers\n')
					for feature in self:
						feature.write(outfile)
				elif group == 'ORIGIN' or 'ORIGIN' not in self.groups:
					# should there be spaces after ORIGIN?
					outfile.write('ORIGIN      ')
					i = 0
					dna = textwrap.wrap(self.dna, 10)
					for block in dna:
						if(i%60 == 0):
							outfile.write('\n')
							outfile.write(str(i+1).rjust(9))
							outfile.write(' ')
							outfile.write(block.lower())
						else:
							outfile.write(' ')
							outfile.write(block.lower())
						i += 10
				elif group == 'BASE':
					for value in values:
						outfile.write(group)
						outfile.write(' ')
						outfile.write(value)
				else:
					outfile.write(group.ljust(12))
					outfile.write(value)
		outfile.write('\n')
		outfile.write('//')
		outfile.write('\n')

	def last(self, n, codons, strand):
		# this needs to be 0-based indexing
		if isinstance(codons, str):
			codons = [codons.lower()]
		codons = [codon.lower() for codon in codons]
		if strand > 0:
			irange = range(n,            -1, -3)
		else:
			irange = range(n, self.length(), +3)

		for i in irange:
			if self.seq(i,i+3,strand) in codons:
				return i
		return None

	def next(self, n, codons, strand):
		if isinstance(codons, str):
			codons = [codons]
		codons = [codon.lower() for codon in codons]
		if strand > 0:
			irange = range(n, self.length(), +3)
		else:
			irange = range(n,            -1, -3)
		for i in irange:
			if self.seq(i,i+3,strand) in codons:
				return i
		return None

	def nearest(self, n, strand, codons):
		_last = self.last(n,strand,codons)
		if not _last:
			_last = 0
		_next = self.next(n,strand,codons)
		if not _next:
			_next = self.length()
		if n - _last < _next - n:
			return _last
		else:
			return _next

	def distance(self, n, strand, codons):
		nearest = self.nearest(n, strand, codons)
		return n - nearest if nearest < n else nearest - n

	def codon_rarity(self, codon=None):
		if not hasattr(self, 'rarity'):
			seen = {-1:dict(), +1:dict()}
			self.rarity = {a+b+c : 0 for a in 'acgt' for b in 'acgt' for c in 'acgt'}
			for feature in self:
				if feature.type == 'CDS':
					for _codon, _loc in zip(feature.codons(), feature.codon_locations()):
						if _codon in self.rarity and _loc not in seen[feature.strand]:
							self.rarity[_codon] += 1
							seen[feature.strand][_loc] = True
		total = sum(self.rarity.values())
		self.rarity = {codon:self.rarity[codon]/total for codon in self.rarity}
		if codon in self.rarity:
			return self.rarity[codon]
		elif codon:
			return None
		else:
			return self.rarity

	def slice(self, left, right):
		if left > right:
			left,right = right+1,left+1
			self.strand = -1
		self.dna = self.seq(left,right)
		for feature in list(self.keys()):
			if feature.right() - 1 < left:
				del self[feature]
			elif feature.left() > right:
				del self[feature]
			else:
				# whew there is a lot going on here
				f0 = lambda x : int(x.replace('<','').replace('>',''))
				f1 = lambda x : '<1' if f0(x) - left < 1 else ('>'+str(self.length()) if f0(x) - left > self.length() else f0(x) - left)
				f2 = lambda x : str(f1(x))
				feature.pairs = rmap(f2, feature.pairs)
		return self
		



		

