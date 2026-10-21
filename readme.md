# Raqun Lab

**Raqun Lab** is a comprehensive benchmarking program for comparing model matching algorithms. It provides an end-to-end framework for loading datasets, executing algorithms, comparing methods, and visualizing results. 

This repository is a fork of [JohansonB/PyQuN](https://github.com/JohansonB/PyQuN) that packages the lab as an installable Python package (`pyqun`) and fixes several bugs; [CHANGELOG.md](./CHANGELOG.md) lists what changed. RaQuN itself is the algorithm of Schultheiß et al. (see [References](#references)).

---

## **Installation**

Requires Python 3.12 or newer. Install a tagged release from GitHub:

```bash
pip install "pyqun @ git+https://github.com/RomanBoegli/PyQuN@v1.0.1"
# or, in a uv project
uv add "pyqun @ git+https://github.com/RomanBoegli/PyQuN@v1.0.1"
```

The distribution is called `pyqun`; the import package is `RaQuN_Lab`. The core (data model, RaQuN, candidate search, matching) needs only numpy and scikit-learn. Optional extras add the rest:

| Extra        | Adds                  | Needed for                                     |
|--------------|-----------------------|------------------------------------------------|
| `evaluation` | matplotlib, tabulate  | `evaluation.XYPlot`, `evaluation.ResultView`   |
| `lsh`        | nearpy                | the `LSHKNN` candidate search                  |
| `yaml`       | pyyaml                | `utils.YamlParser`                             |

For example `pip install "pyqun[evaluation,lsh] @ git+https://github.com/RomanBoegli/PyQuN@v1.0.1"`.

## **Matching Models with RaQuN**

```python
from RaQuN_Lab.datamodel.modelset.Element import Element
from RaQuN_Lab.datamodel.modelset.Model import Model
from RaQuN_Lab.datamodel.modelset.ModelSet import ModelSet
from RaQuN_Lab.datamodel.modelset.attribute.StringAttribute import StringAttribute
from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN

# Models and elements are identified by their ids alone, so keep the ids disjoint
crm, shop = Model(0), Model(1)
crm.add_element(Element(10, {StringAttribute("name"), StringAttribute("email")}, name="Customer"))
crm.add_element(Element(11, {StringAttribute("street"), StringAttribute("city")}, name="Address"))
shop.add_element(Element(20, {StringAttribute("name"), StringAttribute("email"), StringAttribute("phone")}, name="Client"))
shop.add_element(Element(21, {StringAttribute("sku"), StringAttribute("price")}, name="Product"))

matching, stopwatch = VanillaRaQuN("raqun").match(ModelSet({crm, shop}))
for match in sorted(matching, key=lambda m: min(e.get_id() for e in m)):
    print(sorted(e.get_name() for e in match))
# ['Client', 'Customer']
# ['Address']
# ['Product']
```

Every match holds at most one element per model. For the same input, RaQuN returns the same matching in every run, whatever `PYTHONHASHSEED` is. The exceptions are the variants that use randomness by design: `LSHKNN` (random projections), `SVDReduction` (randomized SVD) and `ExploringGreedyMatching` (shuffles candidates).

---

## **Key Features**

1. **Dataloader**
   - Efficiently loads and preprocesses datasets for benchmarking.
2. **Experiment Environment**
   - Streamlined setup for executing model matching algorithms.
3. **Comparison Framework**
   - Enables quantitative and qualitative comparison of algorithm outputs.
4. **Visualization Tools**
   - Visualize matches and comparative results of different algorithms.
5. **RaQuN Algorithm Variants**
  - Implements several variants of the RaQuN model matching algorithm.
  - Introduces a novel variation that applies dimensionality reduction as a preprocessing step, combining the benefits of both high-dimensional and low-dimensional vector encodings.

Detailed descriptions of each component are available in the respective subdirectories:

- [Dataloader](./RaQuN_Lab/dataloading)
- [Experiment Environment & Comparison Framework ](./RaQuN_Lab/experiment)
- [Visualization Tools ](./RaQuN_Lab/evaluation)

---

## **Example Use Case**

Below is an example illustrating the use of Raqun Lab to set up and execute experiments. [main.py](./main.py) contains it; run it from the repository root of a clone, since the lab reads its datasets from `MetaData/Datasets` and writes experiments and results under `MetaData/`. It needs the `evaluation` and `lsh` extras (`uv run --all-extras python main.py`).

```python
if __name__ == "__main__":
    # Create the matching algorithms
    low_dim = VanillaRaQuN("2D-raqun")
    high_dim = VanillaRaQuN("high_dim_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer()))
    bfknn = VanillaRaQuN("bfknn_raqun", candidate_search=NNCandidateSearch(knn=BFKNN()))
    svd_k10 = VanillaRaQuN("svd_k10_raqun",
                          candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), reduction=SVDReduction()))
    svd_k50 = VanillaRaQuN("svd_k50_raqun",
                          candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), reduction=SVDReduction(50)))
    lsh = VanillaRaQuN("LSH_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), knn=LSHKNN()))
    lsh_svd = VanillaRaQuN("LSH_SVD_raqun", candidate_search=NNCandidateSearch(vectorizer=ZeroOneVectorizer(), knn=LSHKNN(),
                                                                              reduction=SVDReduction(50)))

    # Initialize the experiments
    match_all = DoMatching("do_matching_all", 5)
    vary_size = VarySize(0.1, 5,"vary_len",5)

    # Add the experiments to the execution pipeline
    ExperimentManager.add_experiment(match_all)
    ExperimentManager.add_experiment(vary_size)

    # Setup the matchall experiment
    ExperimentManager.add_strategies(match_all,[low_dim,high_dim,bfknn,svd_k10,svd_k50,lsh,lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],match_all)
    ExperimentManager.add_strategies(vary_size, [low_dim, high_dim, bfknn, svd_k10, svd_k50, lsh, lsh_svd])
    ExperimentManager.add_datasets(["hosp", "ppu", "argouml", "bcms", "bcs", "ppu_statem", "random", "randomLoose", "randomTight", "warehouses"],vary_size)

    # Run the experiments
    ExperimentManager.run_unfinished_experiments(ThreadPoolExecutor(max_workers=10))
```



This code snippet demonstrates:
- Creation of various model matching algorithms.
- Configuration of experiments.
- Addition of datasets and strategies to the experiment pipeline.
- Execution of experiments in parallel using a thread pool.

## Development

```bash
uv sync          # core dependencies and pytest
uv run pytest
```

The wheel contains only the `RaQuN_Lab` package; `Data/`, `MetaData/` and `main.py` stay in the repository.

## References

If you use RaQuN Lab in your work, please cite the RaQuN paper:

> Alexander Schultheiß, Paul Maximilian Bittner, Alexander Boll, Lars Grunske, Thomas Thüm, and Timo Kehrer. 2023. RaQuN: a generic and scalable n-way model matching algorithm. *Software and Systems Modeling* 22, 5 (2023), 1495–1517. https://doi.org/10.1007/s10270-022-01062-5

The n-way model matching problem, the NwM algorithm and the weight metric (`WeightMetric`) come from:

> Julia Rubin and Marsha Chechik. 2013. N-way model merging. In *Proceedings of the 2013 9th Joint Meeting on Foundations of Software Engineering (ESEC/FSE 2013)*. ACM, 301–311. https://doi.org/10.1145/2491411.2491446

How RaQuN's configuration (neighbourhood size, vectorization, similarity and threshold) trades runtime against matching quality, and how to tune it, is studied in (implementation: [SEG-UNIBE/nexus](https://github.com/SEG-UNIBE/nexus)):

> Jonas Spieler, Roman Bögli, Stephan Weiss, and Timo Kehrer. 2027. Taming the Trade-Off: Efficiency vs. Quality in N-Way Model Matching. In *Software and Systems Reuse, Product Lines, and Configuration (VARIABILITY 2026)*. Lecture Notes in Computer Science. Springer, Cham, 339–361. https://doi.org/10.1007/978-3-032-40763-4_19

```bibtex
@article{schultheiss2023raqun,
  author    = {Schulthei{\ss}, Alexander and Bittner, Paul Maximilian and Boll, Alexander and Grunske, Lars and Th{\"u}m, Thomas and Kehrer, Timo},
  title     = {{RaQuN}: a generic and scalable n-way model matching algorithm},
  journal   = {Software and Systems Modeling},
  volume    = {22},
  number    = {5},
  pages     = {1495--1517},
  year      = {2023},
  doi       = {10.1007/s10270-022-01062-5}
}

@inproceedings{rubin2013nway,
  author    = {Rubin, Julia and Chechik, Marsha},
  title     = {N-way model merging},
  booktitle = {Proceedings of the 2013 9th Joint Meeting on Foundations of Software Engineering (ESEC/FSE 2013)},
  pages     = {301--311},
  publisher = {ACM},
  year      = {2013},
  doi       = {10.1145/2491411.2491446}
}

@inproceedings{spieler2027taming,
  author    = {Spieler, Jonas and B{\"o}gli, Roman and Weiss, Stephan and Kehrer, Timo},
  title     = {Taming the Trade-Off: Efficiency vs. Quality in {N}-Way Model Matching},
  booktitle = {Software and Systems Reuse, Product Lines, and Configuration (VARIABILITY 2026)},
  editor    = {Perrouin, Gilles and Berger, Thorsten and Papadopoulos, George A. and Schaefer, Ina and Fortz, Sophie},
  series    = {Lecture Notes in Computer Science},
  pages     = {339--361},
  publisher = {Springer Nature Switzerland},
  address   = {Cham},
  year      = {2027},
  doi       = {10.1007/978-3-032-40763-4_19}
}
```

RaQuN Lab was written in [JohansonB/PyQuN](https://github.com/JohansonB/PyQuN); this fork builds on it.

## License

MIT, see [LICENSE](./LICENSE).