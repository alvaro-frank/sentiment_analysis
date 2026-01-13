EPOCHS = 5
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

setup:
	$(MKVENV)
	$(PY) -m pip install -r requirements.txt

clean:
	$(RMVENV)

train:
	$(PY) src/train.py --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_words $(MAX_WORDS) $(if $(NROWS),--nrows $(NROWS),)

predict:
	$(PY) src/predict.py --text "$(TEXT)"

evaluate:
	$(PY) src/evaluate.py --nrows $(EVAL_ROWS)

mlflow:
	$(VENV_BIN)\mlflow ui --port $(PORT)

all: clean setup train predict evaluate mlflow