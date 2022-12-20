from genbank.locus import Locus
from genbank_template.feature import Feature

class Locus(Locus, feature=Feature):
	def foo(self):
		'''this adds a custom definition to the Locus class'''
		return 'bar'



