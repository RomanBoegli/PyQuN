# Changelog

## 1.0.0

The first packaged release of this fork. It builds on commit `b117853` of
[RomanBoegli/PyQuN](https://github.com/RomanBoegli/PyQuN), a fork of [JohansonB/PyQuN](https://github.com/JohansonB/PyQuN).

### Changes the matchings RaQuN produces

- **Candidates are sorted by similarity.** `MatchCandidates.sort` called `sorted(...)` and discarded the result, so the
  greedy matching merged candidates in the order the candidate search found them, not by descending similarity as RaQuN
  intends. It now sorts in place, by descending similarity, with ties broken by the sorted element ids of the candidate.
  **RaQuN's matchings therefore differ from those of earlier commits**, for every vectorizer and KNN.
  `ExploringGreedyMatching` now shuffles the top of the similarity order, where before it shuffled the order in which
  candidates were found; it merges via the new `GreedyMatching.merge_candidates`, so a second sort does not undo the
  shuffle.
- **Deterministic results.** The same input now gives the same matching whatever `PYTHONHASHSEED` is:
  - `NNCandidateSearch` and `VectorizedModelSet` walk the elements in element id order instead of set order.
  - `BFKNN` walks models and elements in id order, so equally similar neighbours keep a fixed order.
  - `ZeroOneVectorizer` numbers the attributes in sorted order instead of set order (string hashing).
  - The sort above breaks ties by element ids.

  Randomness by design stays: `LSHKNN` (random projections), `SVDReduction` (randomized SVD), `ExploringGreedyMatching`
  (shuffles candidates) and `RandomMatching` without a seed.

### Packaging

- `pyproject.toml` (hatchling); the distribution is `pyqun`, the import package stays `RaQuN_Lab`. The code moved into
  the folder `RaQuN_Lab/`, so no import changed. Python 3.12 or newer.
- Core dependencies: numpy and scikit-learn (scipy comes with it). Optional extras: `evaluation` (matplotlib, tabulate),
  `lsh` (nearpy), `yaml` (pyyaml). psutil is no longer needed.
- The wheel contains only `RaQuN_Lab`; `Data/`, `MetaData/` and `Main.py` stay in the repository.
- Tests in `tests/` (pytest).

### Fixed

- `Stopwatch` looked up `threading.get_ident()` among psutil's OS thread ids, found nothing and raised `TypeError` on
  the first timer, so every `VanillaRaQuN.match` failed. It now measures CPU time with `time.thread_time()` and
  wall-clock time with `time.perf_counter()` (new: `TimingStruct.wall_elapsed`, `Stopwatch.get_wall_time`). `merge`
  tolerates timers that were never stopped and copies the timers it takes over.
- Default arguments were objects created once at import, so all instances shared them: `VanillaRaQuN`,
  `NNCandidateSearch` (two candidate searches shared one fitted `TreeKNN`), `GreedyMatching`, `ExploringGreedyMatching`,
  `BeamSearchGreedyMatching`, `JaccardIndex`, `WeightMetric`, `BFKNN`, `SimilarityScore`, `EGRaqunPop`,
  `GeneticStrategy`, `ResultView`, `NormalizedResultView` and `Match` (a shared `set()`). Each instance now creates its
  own.
- `ZeroOneVectorizer.innit` kept the attribute map of the previous model set while restarting its counter, so a reused
  vectorizer mixed up indices. It now starts from an empty map.
- `TreeKNN` created the exception for an unknown `tree_type` without raising it; it now raises `ValueError`. It asked
  the tree for more neighbours than the model set has elements, which scikit-learn rejects; the neighbourhood is now
  clamped to the number of elements.
- `Matching.get_match_by_element` scanned every match, so the greedy matching took quadratic time in the number of
  elements. `Matching` keeps an element-to-match index, updated by `add_match` and `merge_matches` and rebuilt if
  `matches` is changed from outside. `Match` gained `__contains__`. About 11× faster on 4,000 elements.
- `MatchCandidates.shuffle` replaced the candidates with `None` (the result of `list.extend`).
- `CompressedMatching.decompress` added plain sets instead of `Match` objects, and `Matching.load` called it without
  its arguments. **API change:** `Matching.load(path, dataset_path, strategy)`.
- `VectorizedModelSet` called `vectorizer.innit` before checking the vectorizer for `None`.
- `experiment.ResultIterator` imported `Iterator` from `collections`, which fails on Python 3.10 and newer; it now
  imports it from `collections.abc`.
