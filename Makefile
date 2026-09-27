# One-command entry points. Each target is a thin wrapper around a script,
# so everything also works without make.

PYTHON ?= python3

.PHONY: install reproduce synthetic check test lint

install:      ## install the package with the real-data and development extras
	$(PYTHON) -m pip install -e ".[real,dev]"
	$(PYTHON) -m spacy download en_core_web_sm

reproduce:    ## every result, table, figure and number, then compare with the commit
	$(PYTHON) scripts/reproduce.py

synthetic:    ## Experiments 1-12 only (NumPy only)
	$(PYTHON) scripts/reproduce.py --synthetic

check:        ## compare the outputs on disk with the committed versions
	$(PYTHON) scripts/reproduce.py --check

test:         ## unit tests and the invariant verification
	$(PYTHON) -m pytest -q

lint:
	ruff check .
