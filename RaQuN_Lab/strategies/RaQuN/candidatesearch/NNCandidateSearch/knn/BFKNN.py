from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.similarities.match.WeightMetric import WeightMetric
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.KNN import KNN


class BFKNN(KNN):
    def __init__(self, similarity:'Similarity' = None) -> None:
        self.similarity = similarity if similarity is not None else WeightMetric()
        self.vm_set = None


    def set_data(self, vm_set):
        self.vm_set =  vm_set

    def get_neighbours(self, element, size):
        ms = self.vm_set.get_model_set()
        sim = []
        # models and elements in id order, so equally similar neighbours keep a fixed order in the stable sort below
        for m in sorted(ms, key=lambda m: m.get_id()):
            if m.get_id() == element.get_model_id():
                continue
            for ele2 in sorted(m, key=lambda e: e.get_id()):
                sim.append((ele2,self.similarity.similarity(Match({element,ele2},do_check=False))))
        sim.sort(key=lambda x : x[1],reverse=True)
        return [x[0] for x in sim[:size]]
