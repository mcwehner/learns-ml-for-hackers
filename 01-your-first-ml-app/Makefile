models	:= dnn_model linear_model

all: $(models)

clean:
	rm -rf $(models)

$(models):
	python train_$@.py

.PHONY: all clean
