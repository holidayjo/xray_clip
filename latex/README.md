# Paper draft (IEEE journal format, e.g. IEEE J-BHI)

    latex/
    ├── main.tex              the paper (IEEEtran journal class)
    ├── references.bib        only papers we actually used (details checked against the PDFs)
    ├── tables/               GENERATED: tab_*.tex and numbers.tex (\newcommand values used in the text)
    ├── figures/              copied from runs/freq_v1/results/ (notebook 7, steps 3 and 5)
    ├── scripts/make_tables.py  builds tables/ from the result files
    └── IEEEtran.cls / .bst   from CTAN (not installed system-wide)

Rebuild after the results change:

    /home/hj/anaconda3/envs/hj/bin/python latex/scripts/make_tables.py
    cp runs/freq_v1/results/relationship{1,2}_exact_vs_stem.png latex/figures/
    cd latex && latexmk -pdf main.tex

Never type a number into main.tex: add it to `numbers.tex` via `make_tables.py`, so the text and
the tables always match the data.
