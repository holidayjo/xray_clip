"""Helpers for 6_frequency_predicts_adaptation.ipynb.

One module per stage of the notebook:
    mentions   section 2      how often each concept is named in the training reports
    data       section 3      test / support / selection splits and their labels
    clip_eval  sections 4-6   checkpoint ranking, feature encoding, zero-shot
    fewshot    section 8      SS-Text and the K sweep
    laws       sections 7-12  bootstrap, Law 1 / Law 2 fits, crossing point, policies

Shared with the rest of the repo: utils/config.py (load_experiment, show) loads
config/freq_v1.yaml, and utils/plot.py (two_laws) draws the section-10 figure.

`clip_eval`, `fewshot` and `data` import CheXzero modules, so CheXzero/ must be on
sys.path before they are imported (the notebook setup cell does this).
"""
