"""Concept frequency exactly as Udandarao et al. (2024) count it in captions, applied to our
training impressions. Run with spaCy 3.7.2 (their requirements.txt), in its own environment:

    ~/.venvs/spacy372/bin/python utils/freq/udandarao_count.py \
        --impressions CheXzero/data/mimic_impressions.csv --concepts "Pleural Effusion" ... --out counts.csv

Ported from github.com/bethgelab/frequency_determines_performance (commit 11ef009):
  index  src/text_search_inverted_index_get_word_dictionaries.py   create_unigram_dictionary_spacy
         every caption parsed with en_core_web_lg; for each NOUN/PROPN token keep its lemma with
         non-letters removed (case kept); record which captions contain each lemma.
  query  src/text_search_matches_inverted_index.py                  search_in_full_index
         class name lower-cased, split on spaces/underscores, each word lemmatized with
         en_core_web_sm; count = captions containing ALL the words. Words that never occur as a
         noun are SKIPPED (not treated as zero), as in their code.
No synonyms and no negation: "no pneumothorax" counts as a pneumothorax mention.

Only difference in mechanics: identical captions are parsed once and mapped back to every row
that holds them. spaCy parses each text independently, so the counts are unchanged.
"""
import argparse
import re

import pandas as pd
import spacy


def caption_nouns(doc):
    """Their create_unigram_dictionary_spacy, for one caption."""
    return set(re.sub(r"[^A-Za-z ]", "", t.lemma_) for t in doc if t.pos_ in ["NOUN", "PROPN"])


def build_index(texts, batch_size, n_process):
    """{noun lemma: set of row numbers}. Unique texts are parsed once."""
    nlp = spacy.load("en_core_web_lg")
    uniq = pd.unique(texts)
    rows_of = pd.Series(range(len(texts))).groupby(texts.values).apply(list).to_dict()
    index = {}
    for text, doc in zip(uniq, nlp.pipe(uniq, batch_size=batch_size, n_process=n_process)):
        for noun in caption_nouns(doc):
            index.setdefault(noun, set()).update(rows_of[text])
    return index


def query_terms(classnames):
    """Their query processing: lower-case, split on [_ ], lemmatize each word (en_core_web_sm)."""
    nlp = spacy.load("en_core_web_sm")
    processed = [re.split(r"[_\s]+", c.lower()) for c in classnames]
    flat = set(w for words in processed for w in words)
    lemma = {w: tok.lemma_ for w in flat for tok in nlp(w)}
    return [[lemma[w] for w in words] for words in processed]


def count(index, terms):
    """Their search_for_text: intersect the rows of every term present in the index."""
    sets = [index[t] for t in terms if t in index]
    return len(set.intersection(*sets)) if sets else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--impressions", required=True)
    ap.add_argument("--concepts", nargs="+", required=True, help="class names as used in the prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=1000)
    ap.add_argument("--n_process", type=int, default=8)
    a = ap.parse_args()

    texts = pd.read_csv(a.impressions)["impression"].fillna("").astype(str)
    print(f"{len(texts):,} training rows, {texts.nunique():,} unique texts; parsing ...", flush=True)
    index = build_index(texts, a.batch_size, a.n_process)
    rows = []
    for name, terms in zip(a.concepts, query_terms(a.concepts)):
        used = [t for t in terms if t in index]
        rows.append({"concept": name, "query_lemmas": " ".join(terms), "lemmas_found": " ".join(used),
                     "count": count(index, terms), "n_rows": len(texts)})
    out = pd.DataFrame(rows)
    out["udandarao_%"] = 100 * out["count"] / out["n_rows"]
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
