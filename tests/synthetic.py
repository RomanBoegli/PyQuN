"""Small synthetic model sets for the tests."""

import itertools
import random

from RaQuN_Lab.datamodel.modelset.Element import Element
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.datamodel.modelset.ModelSet import ModelSet
from RaQuN_Lab.datamodel.modelset.attribute.StringAttribute import StringAttribute


def element(ele_id: int, model_id: int, *attrs: str) -> Element:
    ele = Element(ele_id, {StringAttribute(a) for a in attrs})
    ele.set_model_id(model_id)
    return ele


def model_set(n_models: int = 4, n_elements: int = 25, n_concepts: int = 30, seed: int = 0) -> ModelSet:
    """n_models models of n_elements elements each. Every element is a noisy copy of one of n_concepts attribute sets,
    so elements of different models overlap and many candidates tie on similarity. Models and elements get disjoint
    ids, as the loaders give them."""
    rng = random.Random(seed)
    vocab = [f"attr{i}" for i in range(60)]
    concepts = [rng.sample(vocab, rng.randint(2, 6)) for _ in range(n_concepts)]
    ids = itertools.count()
    models = set()
    for _ in range(n_models):
        model = Model(next(ids))
        for _ in range(n_elements):
            concept = rng.choice(concepts)
            attrs = {a for a in concept if rng.random() < 0.8} or {concept[0]}
            if rng.random() < 0.3:
                attrs.add(rng.choice(vocab))
            model.add_element(Element(next(ids), {StringAttribute(a) for a in attrs}))
        models.add(model)
    return ModelSet(models)


def groups(matching) -> list[list[int]]:
    """A matching as sorted lists of element ids, comparable across runs."""
    return sorted(sorted(e.get_id() for e in m) for m in matching)
