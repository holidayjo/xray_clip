"""
Config loading for the whole repo (YOLO-style).

    config/datasets.yaml   WHERE the datasets are: one root, every file relative to it
    config/<experiment>.yaml   WHAT to run; points to datasets.yaml with `datasets:`

    load_datasets(path)    datasets.yaml -> {"root", "derived", "mimic": {...}, "chexpert": {...}, "nih": {...}}
    load_experiment(path)  experiment yaml + cfg["files"] = absolute paths of every input file
    missing_files(cfg)     every input file that does not exist
    show(cfg, keys)        print a config as readable YAML

Relative paths in a YAML are resolved against the repository root, so it doesn't matter
where Python was started from.
"""

import os
import pathlib

import yaml

REPO = pathlib.Path(__file__).resolve().parent.parent
DATASETS = ("mimic", "chexpert", "nih")


def repo_path(p):
    """Absolute path; relative paths are taken from the repository root."""
    p = pathlib.Path(p)
    return p if p.is_absolute() else REPO / p


def load_yaml(path):
    with open(repo_path(path)) as f:
        return yaml.safe_load(f)


def resolve_section(section):
    """{'path': base, 'a': 'x.csv'} -> {'a': 'base/x.csv'}"""
    s    = dict(section)
    base = s.pop("path")
    return {k: os.path.join(base, v) for k, v in s.items()}


def load_datasets(path="config/datasets.yaml"):
    """Dataset locations as absolute path strings. Non-path entries (e.g. the list of
    download URLs) are passed through unchanged. CXR_DATA_ROOT overrides `root`."""
    cfg  = load_yaml(path)
    root = pathlib.Path(os.environ.get("CXR_DATA_ROOT", cfg["root"]))
    out  = {"root": str(root), "derived": str(repo_path(cfg["derived"]))}
    for ds in DATASETS:
        out[ds] = {k: str(root / v) if isinstance(v, str) else v for k, v in cfg[ds].items()}
    return out


def load_experiment(path):
    """Experiment YAML, plus cfg['files'][section][key] = absolute path of every input:
    the datasets from `datasets:`, and each extra section that has a `path:` base."""
    cfg = load_yaml(path)
    cfg["config_file"] = str(repo_path(path))
    ds = load_datasets(cfg["datasets"])
    cfg["files"] = {name: ds[name] for name in DATASETS}
    for name, section in cfg.items():
        if isinstance(section, dict) and "path" in section:
            cfg["files"][name] = resolve_section(section)
    if "synonyms" in cfg:
        cfg["synonyms"] = str(repo_path(cfg["synonyms"]))
    return cfg


def missing_files(cfg):
    """Every input path in cfg['files'] that does not exist on disk."""
    return [p for sec in cfg["files"].values() for p in sec.values()
            if isinstance(p, str) and not os.path.exists(p)]


def show(cfg, keys=None):
    """Print the config as YAML (all of it, or only `keys`): mappings as indented blocks,
    lists on one line."""
    class Dumper(yaml.SafeDumper):
        pass
    Dumper.add_representer(list, lambda d, v: d.represent_sequence("tag:yaml.org,2002:seq", v, flow_style=True))
    part = cfg if keys is None else {k: cfg[k] for k in keys}
    print(yaml.dump(part, Dumper=Dumper, sort_keys=False, width=1000).rstrip())
