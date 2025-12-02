EPOCHS = 5
BATCH_SIZE = 32
MAX_WORDS = 5000
NROWS = 1000
EVAL_ROWS = 1000
TEXT = Stock market rallies on positive economic news
PORT = 5000

PYTHON := py -3.10
VENV_BIN := .venv\Scripts
PY  := $(VENV_BIN)\python.exe
MKVENV := if not exist .venv ( $(PYTHON) -m venv .venv )

setup:
	$(MKVENV)
	$(PY) -m pip install -r requirements.txt

train:
	$(SET_PYTHONPATH) $(PY) src/train.py --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_words $(MAX_WORDS) --nrows $(NROWS)

predict:
	$(SET_PYTHONPATH) $(PY) src/predict.py --text "$(TEXT)"

evaluate:
	$(SET_PYTHONPATH) $(PY) src/evaluate.py --nrows $(EVAL_ROWS)

mlflow:
	$(VENV_BIN)\mlflow ui --port $(PORT)

all: setup train predict evaluate mlflow