models	:= dnn_model linear_model

.PHONY: all
all: $(models)

$(models):
	python train_$@.py

.PHONY: clean
clean:
	rm -rf $(models)
