"""MatchCandidates.sort orders by descending similarity with deterministic ties, so greedy matching does not depend
on the order in which candidates were found."""

import random

from synthetic import element, groups, model_set

from RaQuN_Lab.datamodel.matching.Match import Match
from RaQuN_Lab.datamodel.matching.MatchCandidates import MatchCandidates
from RaQuN_Lab.datamodel.matching.similarities.match.JaccardIndex import JaccardIndex
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.ZeroOneVectorization import \
    ZeroOneVectorizer
from RaQuN_Lab.strategies.RaQuN.matching.ExploreingGreedyMatching import ExploringGreedyMatching
from RaQuN_Lab.strategies.RaQuN.matching.GreedyMatching import GreedyMatching


def _ids(candidates):
    return [sorted(e.get_id() for e in m) for m in candidates]


def test_sort_orders_by_descending_similarity_with_deterministic_ties():
    a = element(1, 0, "x", "y")
    b = element(2, 1, "x", "y")
    c = element(3, 2, "x", "z")
    d = element(4, 3, "q")
    # Jaccard: (a, b) = 1, (a, c) = (b, c) = 1/3 (a tie), (c, d) = 0
    pairs = [(c, d), (b, c), (a, c), (a, b)]
    for seed in range(5):
        random.Random(seed).shuffle(pairs)
        mc = MatchCandidates()
        mc.add_matches(*[Match({x, y}) for x, y in pairs])
        mc.sort(JaccardIndex())
        assert _ids(mc) == [[1, 2], [1, 3], [2, 3], [3, 4]]
        assert mc.is_sorted


def test_greedy_matching_does_not_depend_on_candidate_order():
    ms = model_set(seed=3)
    found, _ = NNCandidateSearch(vectorizer=ZeroOneVectorizer()).candidate_search(ms)
    found = list(found)
    expected = None
    for seed in range(4):
        random.Random(seed).shuffle(found)
        mc = MatchCandidates()
        mc.add_matches(*found)
        result = groups(GreedyMatching().match(ms, mc))
        expected = expected or result
        assert result == expected
    assert any(len(g) > 1 for g in expected)


def test_exploring_greedy_matching_keeps_its_shuffle():
    ms = model_set(seed=3)
    search = NNCandidateSearch(vectorizer=ZeroOneVectorizer())

    def run(strategy):
        return groups(strategy.match(ms, search.candidate_search(ms)[0]))

    greedy = run(GreedyMatching())
    assert run(ExploringGreedyMatching(shuffle_threshold=0)) == greedy
    random.seed(0)
    shuffled = [run(ExploringGreedyMatching(shuffle_threshold=1.0)) for _ in range(5)]
    assert any(s != greedy for s in shuffled)


def test_shuffle_keeps_the_candidates():
    mc = MatchCandidates()
    matches = [Match({element(i, i)}) for i in range(10)]
    mc.add_matches(*matches)
    mc.shuffle(0.5)
    assert isinstance(mc.matches, list) and set(mc.matches) == set(matches)
    assert mc.matches[5:] == matches[5:]
