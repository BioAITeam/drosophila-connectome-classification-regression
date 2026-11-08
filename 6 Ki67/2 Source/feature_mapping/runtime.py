"""Portable release paths and shared training implementation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").is_file())
ARCHIVE = "6 Ki67/5 Outputs/phikon-feature-mapping"
WORKSPACE = "6 Ki67/6 Workspace/phikon-feature-mapping"

def local(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Use repository-relative paths")
    return REPO / path

def relative(path):
    return Path(path).absolute().relative_to(REPO).as_posix()

def writable(path):
    path = Path(path)
    path.resolve().relative_to((REPO / "6 Ki67/6 Workspace").resolve())
    return path

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)

def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()

def implementation_hashes():
    files = sorted(HERE.glob("*.py"))
    package = HERE.parent
    files += [package / (name + ".py") for name in
              ("__init__", "common", "training", "sequence", "data", "features", "encoders", "graphs", "evaluation", "tasks")]
    files += [REPO / "3 Connectome/flybrain" / (name + ".py")
              for name in ("__init__", "model", "paths")]
    return {relative(path): sha(path) for path in files}

def activate():
    environment = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[key] = "4"
    for library in Path(sys.prefix).glob("lib/python*/site-packages/nvidia/nvjitlink/lib"):
        if str(library) not in environment.get("LD_LIBRARY_PATH", "").split(":"):
            environment["LD_LIBRARY_PATH"] = str(library) + ":" + environment.get("LD_LIBRARY_PATH", "")
    if environment.get("LD_LIBRARY_PATH") != os.environ.get("LD_LIBRARY_PATH"):
        os.execve(sys.executable, [sys.executable, "-B", *sys.argv], environment)
    os.environ.update(environment)
    os.environ["FLYBRAIN_REPO"] = str(REPO)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(REPO / "3 Connectome"))
    package = HERE.parent
    spec = importlib.util.spec_from_file_location("ki67", package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["ki67"] = module
    spec.loader.exec_module(module)
