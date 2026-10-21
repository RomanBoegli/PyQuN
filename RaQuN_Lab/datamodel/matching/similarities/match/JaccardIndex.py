from RaQuN_Lab.datamodel.matching.similarities.match.Similarity import Similarity
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ShouldMatchStrategy import ShouldMatchStrategy
from RaQuN_Lab.datamodel.matching.similarities.match.shouldmatch.ThresholdSMStrategy import ThresholdSMStrategy


class JaccardIndex(Similarity):
    def __init__(self, strategy:ShouldMatchStrategy = None) -> None:
        super().__init__(strategy if strategy is not None else ThresholdSMStrategy())


    def innit(self, m_s: 'Modelset'):
        pass

    def similarity(self, match):
        sets = [set(element) for element in match]
        intersection = set.intersection(*sets)
        union = set.union(*sets)
        return len(intersection)/len(union)

    def should_match(self, matches, elements=None):
        return self.strategy.should_match(self,matches,elements)
