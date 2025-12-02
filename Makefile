EPOCHS = 5
BATCH_SIZE = 32
MAX_WORDS = 5000
NROWS = 50000
TEXT = Stock market rallies on positive economic news

setup:
	python -m pip install --upgrade pip
	python -m pip install -r requirements.txt

train:
	python src/train.py --epochs $(EPOCHS) --batch_size $(BATCH_SIZE) --max_words $(MAX_WORDS) --nrows $(NROWS)

predict:
	python src/predict.py --text "$(TEXT)"

all: setup train predict