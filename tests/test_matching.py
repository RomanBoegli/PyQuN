"""RaQuN's matchings are valid (at most one element per model, every element in exactly one match), and the fixes to
the matching data structures, the KNN, the vectorizer and the defaults hold."""

import pytest
from synthetic import element, groups, model_set

from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.Matching import Matching
from RaQuN_Lab.datamodel.matching.similarities.match.JaccardIndex import JaccardIndex
from RaQuN_Lab.datamodel.matching.similarities.match.WeightMetric import WeightMetric
from RaQuN_Lab.datamodel.matching.similarities.matching.SimilarityScore import SimilarityScore
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.datamodel.modelset.ModelSet import ModelSet
from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.BFKNN import BFKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.TreeKNN import TreeKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.ZeroOneVectorization import \
    ZeroOneVectorizer
from RaQuN_Lab.strategies.RaQuN.matching.GreedyMatching import GreedyMatching

VARIANTS = {
    "default": lambda: VanillaRaQuN("default"),
    "zero_one": lambda: VanillaRaQuN("zero_one", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer())),
    "ball_tree": lambda: VanillaRaQuN("ball_tree", candidate_search=NNCandidateSearch(knn=TreeKNN("BallTree"))),
    "bfknn": lambda: VanillaRaQuN("bfknn", candidate_search=NNCandidateSearch(knn=BFKNN())),
}


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_match_holds_at_most_one_element_per_model(variant):
    ms = model_set(n_models=5, seed=2)
    matching, stopwatch = VARIANTS[variant]().timed_match(ms)
    for m in matching:
        model_ids = [e.get_model_id() for e in m]
        assert len(model_ids) == len(set(model_ids))
    # a partition of the elements
    ids = [i for g in groups(matching) for i in g]
    assert sorted(ids) == sorted(e.get_id() for e in ms.get_elements())
    assert any(len(m) > 1 for m in matching)
    assert stopwatch.get_time("matching") >= 0 and stopwatch.get_time("candidate_search") >= 0


def test_get_match_by_element_follows_merges_and_outside_changes():
    ms = model_set(seed=5)
    matching, _ = VanillaRaQuN("r").match(ms)
    for ele in ms.get_elements():
        m = matching.get_match_by_element(ele)
        assert ele in m and m in matching.get_matches()
    # replacing the set from outside: the index notices and rebuilds
    ele = next(iter(ms.get_elements()))
    old = matching.get_match_by_element(ele)
    single = Match({ele})
    matching.matches = {m for m in matching.matches if m != old} | {single} | {
        Match(set(old.get_elements()) - {ele}, do_check=False)}
    assert matching.get_match_by_element(ele) == single
    with pytest.raises(ValueError):
        matching.get_match_by_element(element(10**6, 0, "x"))


def test_no_shared_default_objects():
    a, b = VanillaRaQuN("a"), VanillaRaQuN("b")
    assert a.search is not b.search
    assert a.search.knn is not b.search.knn and a.search.vectorizer is not b.search.vectorizer
    assert a.match_strat is not b.match_strat and a.match_strat.similarity is not b.match_strat.similarity
    assert JaccardIndex().strategy is not JaccardIndex().strategy
    assert WeightMetric().strategy is not WeightMetric().strategy
    assert BFKNN().similarity is not BFKNN().similarity
    assert SimilarityScore().similarity is not SimilarityScore().similarity
    assert GreedyMatching().similarity is not GreedyMatching().similarity
    assert Match().get_elements() is not Match().get_elements()


def test_zero_one_vectorizer_reuse_starts_fresh_and_numbers_in_sorted_order():
    vec = ZeroOneVectorizer()
    vec.innit(model_set(seed=1))
    small = ModelSet({Model(0, {element(1, 0, "b", "a")}), Model(2, {element(3, 2, "c")})})
    vec.innit(small)
    assert {a.value: i for a, i in vec.attr_index_map.items()} == {"a": 0, "b": 1, "c": 2}
    assert vec.dim() == 3


def test_tree_knn_clamps_the_neighbourhood_and_rejects_unknown_trees():
    small = ModelSet({Model(0, {element(1, 0, "x", "y")}), Model(2, {element(3, 2, "x", "y")}),
                      Model(4, {element(5, 4, "z")})})
    matching, _ = VanillaRaQuN("small", candidate_search=NNCandidateSearch(neighbourhood_size=10)).match(small)
    assert groups(matching) == [[1, 3], [5]]
    with pytest.raises(ValueError):
        TreeKNN(tree_type="Quadtree").set_data(small.vectorize(ZeroOneVectorizer()))


def test_store_and_load_round_trip(tmp_path):
    # CSVLoader format: "model_id,element_id,element_name,first attribute" then the other attributes, ";"-separated
    csv = tmp_path / "tiny.csv"
    csv.write_text("0,0,a1,x;y;z\n0,1,a2,p;q\n1,2,b1,x;y\n1,3,b2,p;q;r\n2,4,c1,x;y;z\n")
    strategy = VanillaRaQuN("tiny", candidate_search=NNCandidateSearch(3, vectorizer=ZeroOneVectorizer()))
    ms = strategy.get_data_loader(str(csv)).read_file(str(csv)).get_data_model()
    matching, _ = strategy.match(ms)
    matching.store(str(tmp_path / "matching"))
    loaded = Matching.load(str(tmp_path / "matching"), str(csv), strategy)
    assert all(isinstance(m, Match) for m in loaded)
    assert groups(loaded) == groups(matching)
