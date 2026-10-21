"""The same input gives the same matching whatever PYTHONHASHSEED is (string hashing decides set order)."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = """
import json
from synthetic import groups, model_set
from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.NNCandidateSearch import NNCandidateSearch
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.BFKNN import BFKNN
from RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.vectorization.ZeroOneVectorization import \\
    ZeroOneVectorizer

variants = {
    "default": lambda: VanillaRaQuN("default"),
    "zero_one": lambda: VanillaRaQuN("zero_one", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer())),
    "bfknn": lambda: VanillaRaQuN("bfknn", candidate_search=NNCandidateSearch(knn=BFKNN())),
}
print(json.dumps({name: groups(make().match(model_set(seed=1))[0]) for name, make in variants.items()}))
"""


def _run(hash_seed: str) -> dict:
    # run from the tests directory, so the subprocess imports the same RaQuN_Lab as the tests and finds synthetic.py
    tests_dir = Path(__file__).parent
    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True, env=env,
                         cwd=tests_dir).stdout
    return json.loads(out)


def test_same_matching_under_different_hash_seeds():
    first, second = _run("0"), _run("4242")
    assert first == second
    assert all(any(len(g) > 1 for g in matching) for matching in first.values())
