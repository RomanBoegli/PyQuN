from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.matching.GreedyMatching import GreedyMatching
from RaQuN_Lab.strategies.Strategy import Strategy


class VanillaRaQuN(Strategy):

    def __init__(self, name:str, candidate_search: 'CandidateSearch' = None, matching_strategy: 'MatchingStrategy' = None) -> None:
        # defaults are created per instance: a shared NNCandidateSearch would share its fitted KNN tree
        self.search = candidate_search if candidate_search is not None else NNCandidateSearch()
        self.match_strat = matching_strategy if matching_strategy is not None else GreedyMatching()
        super().__init__(name)

    def match(self, data_model: 'ModelSet') -> ['Matching', 'Stopwatch']:
        cans, s = self.search.timed_candidate_search(data_model)
        return self.match_strat.match(data_model, cans), s