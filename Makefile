PYTHON ?= python3

.PHONY: install train test run ci

install:
	$(PYTHON) -m pip install -r requirements.txt

train:
	$(PYTHON) src/train_model.py

test:
	$(PYTHON) -m unittest test_dt.py test_nlp_bilstm.py

run:
	streamlit run app.py

ci: train test
