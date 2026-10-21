
#create a 0-1 vector where each entry of the vetor represents an attribute in the attribute space
#for each element a 0 represents the absence and a 1 the presence of said attribute
import numpy as np

from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.Vectorizer import Vectorizer


class ZeroOneVectorizer(Vectorizer):
    def __init__(self):
        self.attr_index_map = {}
        self.vec_dim = 0

    def innit(self, m_s: 'ModelSet') -> None:
        # map every attribute of the model set to a vector index. The map starts empty, so a reused vectorizer does
        # not mix up model sets, and the attributes are numbered in sorted order, which does not depend on set order
        attributes = set()
        for element in m_s.get_elements():
            attributes.update(element)
        ordered = sorted(attributes, key=lambda a: (type(a).__name__, repr(a.value)))
        self.attr_index_map = {attribute: index for index, attribute in enumerate(ordered)}
        self.vec_dim = len(self.attr_index_map)






    def vectorize(self, element:'Element'):
        vec = np.zeros(self.vec_dim)
        for attr in element:
            vec[self.attr_index_map[attr]] = 1
        return vec

    def dim(self):
        return self.vec_dim

