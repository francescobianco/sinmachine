.PHONY: chat run train hello-world list-models help

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
	python3 trainer.py datasets/hello-world.jsonl --base default --output hello-world --joint

list-models:
	python3 sinmachine.py --list-models

help:
	@echo ""
	@echo "  make chat                            interactive chat (MODEL=default)"
	@echo "  make chat MODEL=sparse               chat with a specific model"
	@echo "  make run Q='...' MODEL=dense         single question"
	@echo "  make train                           train on datasets/sample.jsonl"
	@echo "  make train DATASET=... BASE=... OUTPUT=..."
	@echo "  make hello-world                     train hello-world model"
	@echo "  make list-models                     show available models"
	@echo ""
