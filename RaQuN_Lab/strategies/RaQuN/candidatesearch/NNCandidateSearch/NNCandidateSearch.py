from RaQuN_Lab.datamodel.matching.Match import Match, MatchViolation
from RaQuN_Lab.datamodel.matching.MatchCandidates import MatchCandidates
from RaQuN_Lab.strategies.RaQuN.candidatesearch.CandidateSearch import CandidateSearch
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.TreeKNN import TreeKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.MeanCountVectorization import \
    MeanCountVectorizer


class NNCandidateSearch(CandidateSearch):
    def __init__(self,neighbourhood_size:int = 10, vectorizer:'Vectorizer' = None, knn:'KNN' = None, reduction: 'DimensionalityReduction' = None) -> None:
        # defaults are created per instance: the KNN holds the data it was fitted on
        self.neighbourhood_size = neighbourhood_size
        self.vectorizer = vectorizer if vectorizer is not None else MeanCountVectorizer()
        self.knn = knn if knn is not None else TreeKNN()
        self.reduction = reduction

    def candidate_search(self, model_set):
        mc = MatchCandidates()

        vm_set, s = model_set.timed_vectorize(self.vectorizer, self.reduction)
        s1 = self.knn.timed_set_data(vm_set)
        s1.merge(s)
        # elements in id order, so the candidates do not depend on the iteration order of the element set
        for ele in sorted(model_set.get_elements(), key=lambda e: e.get_id()):
            neighbours = self.knn.get_neighbours(ele,self.neighbourhood_size)
            for neigh in neighbours:
                match = Match()
                try:
                    match = match.add_elements(ele,neigh)
                    mc.add_matches(match)
                except MatchViolation as e:
                    pass
        return mc, s1
