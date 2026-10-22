PYTHON ?= $(shell command -v python3 2>/dev/null || command -v python)
export PYTHONPATH := src

.PHONY: data train test

data:
	$(PYTHON) -m triage.data.build_dataset

train:
	$(PYTHON) -m triage.models.train

test:
	$(PYTHON) -m pytest -q
