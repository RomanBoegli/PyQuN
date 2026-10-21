"""The core imports with numpy and scikit-learn alone; only the modules behind an extra need more."""

import json
import subprocess
import sys
from pathlib import Path

# module -> the extra that provides what it imports
OPTIONAL = {
    "RaQuN_Lab.evaluation.ResultView": "evaluation",
    "RaQuN_Lab.evaluation.XYPlot": "evaluation",
    "RaQuN_Lab.strategies.RaQuN.candidatesearch.NNCandidateSearch.knn.LSHKNN": "lsh",
    "RaQuN_Lab.utils.YamlParser": "yaml",
}

# Imports every module of the package while the optional dependencies (and psutil, no longer needed) are blocked, so
# the test holds even in an environment that has the extras installed
SCRIPT = """
import importlib, importlib.abc, json, pkgutil, sys

BLOCKED = {"matplotlib", "tabulate", "nearpy", "yaml", "psutil"}

class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked")
        return None

sys.meta_path.insert(0, Block())
import RaQuN_Lab
failed = {}
for info in pkgutil.walk_packages(RaQuN_Lab.__path__, "RaQuN_Lab."):
    try:
        importlib.import_module(info.name)
    except ImportError as e:
        failed[info.name] = str(e)
print(json.dumps(failed))
"""


def test_core_imports_without_optional_dependencies():
    # from the tests directory, so "-c" does not put a source checkout ahead of the installed package
    out = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).parent).stdout
    failed = json.loads(out)
    assert set(failed) == set(OPTIONAL), failed


def test_raqun_entry_point_imports():
    from RaQuN_Lab.strategies.RaQuN.RaQuN import VanillaRaQuN

    assert VanillaRaQuN("smoke").get_name() == "smoke"
