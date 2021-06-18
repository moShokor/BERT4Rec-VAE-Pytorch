from .concat import ConcatCombination
from .factored import FactoredCombination
from .sum import SumCombination

COMBINERS = {
    SumCombination.code(): SumCombination,
    ConcatCombination.code(): ConcatCombination,
    FactoredCombination.code(): FactoredCombination
}


def combiners_factory(code, extractors, embedding_size):
    return COMBINERS[code](extractors, embedding_size)
