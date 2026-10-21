import numpy as np


class VectorizedModelSet:
    #takes as input a vectorizer and augments each element with a vector representation of it
    def __init__(self, model_set:'ModelSet', vectorizer:'Vectorizer', dim_reduction: 'DimensionalityReduction' = None) -> None:
        self.dim_reduction = dim_reduction
        self.model_set = model_set
        self.index_id_map = []
        self.id_index_map = {}
        self.vec_mat = None
        if vectorizer is not None:
            vectorizer.innit(model_set)
            # rows in element id order, so the matrix does not depend on the iteration order of the element set
            elements = sorted(self.model_set.get_elements(), key=lambda e: e.get_id())
            self.vec_mat = np.zeros((len(elements), vectorizer.dim()), dtype=np.float64)
            count = 0
            for element in elements:
                self.index_id_map.append(element.get_id())
                self.id_index_map[element] = count
                self.vec_mat[count] = vectorizer.vectorize(element)
                count += 1

        if dim_reduction is not None and self.vec_mat is not None:
            self.vec_mat = dim_reduction.reduce(self.vec_mat)

    def get_model_set(self):
        return self.model_set


    def get_vec_mat(self):
        return self.vec_mat

    def get_vec_index(self, element:'Element') -> int:
        assert element in self.id_index_map
        return self.id_index_map[element]

    def get_ele_by_index(self, index:int) -> 'Element':

        return self.model_set.get_by_id(self.index_id_map[index])


