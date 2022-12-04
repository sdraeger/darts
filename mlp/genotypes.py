from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'dense',
	'relu',
	'sigmoid',
	'tanh',
	'skip_connect'
]
