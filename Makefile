.PHONY: chat run train hello-world test-hello-world simple-sums test-simple-sums list-models help

MODEL      ?= default
DATASET    ?= datasets/sample.jsonl
BASE       ?= default
OUTPUT     ?= trained
RESOLUTION ?= 500

chat:
	python3 sinmachine.py --chat --model $(MODEL)

run:
	python3 sinmachine.py --model $(MODEL) $(Q)

train:
	python3 trainer.py $(DATASET) --base $(BASE) --output $(OUTPUT) --resolution $(RESOLUTION)

hello-world:
	python3 trainer.py datasets/hello-world.jsonl --base dense --output hello-world --joint

test-hello-world:
	python3 sinmachine.py --chat --model hello-world

simple-sums:
	python3 trainer.py datasets/simple-sums.jsonl --base dense --output simple-sums --joint

test-simple-sums:
	python3 sinmachine.py --chat --model simple-sums

list-models:
	python3 sinmachine.py --list-models

help:
	@echo ""
	@echo "  make chat                            interactive chat (MODEL=default)"
	@echo "  make run Q='...' MODEL=dense         single question"
	@echo "  make train                           train on datasets/sample.jsonl"
	@echo "  make hello-world                     train hello-world model"
	@echo "  make test-hello-world                chat with the hello-world model"
	@echo "  make simple-sums                     train simple-sums model"
	@echo "  make test-simple-sums                chat with the simple-sums model"
	@echo "  make list-models                     show available models"
	@echo ""
