PROJECT_NAME=cs229-project
CONDAROOT=~/miniconda3/bin

SHELL=/bin/bash
PYTHON=python
ACTIVATE_CONDA=source $(CONDAROOT)/activate $(PROJECT_NAME)

.PHONY: all load_data train demo profile write_video figures

all: load_data train

demo:
	$(ACTIVATE_CONDA); \
	$(PYTHON) cs229/demo.py 

profile:
	$(ACTIVATE_CONDA); \
	$(PYTHON) cs229/demo.py --no_display

write_video:
	$(ACTIVATE_CONDA); \
	$(PYTHON) cs229/demo.py --no_display --write_video

load_data:
	$(ACTIVATE_CONDA); \
	$(PYTHON) cs229/load_data_id.py; \
	$(PYTHON) cs229/load_data_is_fly.py; \
	$(PYTHON) cs229/load_data_orientation.py; \
	$(PYTHON) cs229/load_data_wing.py

train:
	$(ACTIVATE_CONDA); \
	$(PYTHON) cs229/train_id.py; \
	$(PYTHON) cs229/train_is_fly.py; \
	$(PYTHON) cs229/train_orientation.py; \
	$(PYTHON) cs229/train_wing.py;

figures: train
	dot -Teps output/diagrams/tree_is_fly.dot -o output/diagrams/tree_is_fly.eps
