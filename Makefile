EPOCHS = 15
BATCH_SIZE = 32
MAX_WORDS = 5000
EVAL_ROWS = 1000
TEXT = Stock market rallies on positive economic news
PORT = 5000

PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PY  := $(VENV_BIN)\python.exe
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )
RMVENV := if exist .venv rmdir /s /q .venv
RESUME ?= False

RESUME_FLAG =
ifeq ($(RESUME),True)
	RESUME_FLAG = --resume
endif

setup:
	$(MKVENV)
	$(PY) -m pip install -r requirements.txt
	$(PY) -m dvc pull

pull-data:
	$(PY) -m dvc pull

clean:
	$(RMVENV)

generate-data:
	$(PY) src/generate_dataset.py $(if $(NROWS),--nrows $(NROWS),)

train:
	$(PY) src/train.py $(RESUME_FLAG) --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_words $(MAX_WORDS)

predict:
	set PYTHONPATH=. && $(PY) -m src.predict --text "$(TEXT)"

evaluate:
	$(PY) src/evaluate.py --nrows $(EVAL_ROWS)

mlflow:
	$(VENV_BIN)\mlflow ui --port $(PORT)

unit-test:
	set PYTHONPATH=src && $(PY) -m pytest tests/

all: clean setup unit-test train predict evaluate mlflow