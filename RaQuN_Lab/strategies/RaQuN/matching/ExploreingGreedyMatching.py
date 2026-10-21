from RaQuN_Lab.datamodel.matching.similarities.match.JaccardIndex import JaccardIndex
from RaQuN_Lab.strategies.RaQuN.matching.GreedyMatching import GreedyMatching


class ExploringGreedyMatching(GreedyMatching):
    def __init__(self, similarity: 'Similarity' = None, filter_threshold: float = 0.001, shuffle_threshold : float = 0.2):
        super(ExploringGreedyMatching, self).__init__(similarity, filter_threshold)
        self.shuffle_threshold = shuffle_threshold

    def match(self, model_set: 'ModelSet', candidates: 'MatchCandidates' = None) -> 'Matching':
        candidates.filter(self.similarity, self.filter_threshold)
        candidates.sort(self.similarity)
        candidates.shuffle(self.shuffle_threshold)
        # merge in the shuffled order: GreedyMatching.match would sort again and undo the shuffle
        return self.merge_candidates(model_set,candidates)


