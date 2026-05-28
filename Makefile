.PHONY: chat run train hello-world test-hello-world simple-sums simple-sums-mj simple-sums-stream simple-sums-noend simple-sums-me hello-world-me test-simple-sums list-models help

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
	python3 vocab_align.py --q "1+1=" --a "2" --harmonics 2 --budget 10 --rounds 20 --save simple-sums

simple-sums-mj:
	python3 trainer.py datasets/simple-sums.jsonl --base dense --output simple-sums-mj --multijoint

simple-sums-stream:
	python3 trainer.py datasets/simple-sums.jsonl --base dense --output simple-sums-stream --stream

simple-sums-noend:
	python3 trainer.py datasets/simple-sums-noend.jsonl --base dense --output simple-sums-noend --multijoint

simple-sums-me:
	python3 trainer.py datasets/simple-sums.jsonl --base dense --output simple-sums-me --multijoint --multi-end

hello-world-me:
	python3 trainer.py datasets/hello-world.jsonl --base dense --output hello-world-me --joint --multi-end

test-simple-sums:
	python3 sinmachine.py --chat --model simple-sums

benchmark:
	python3 benchmark.py datasets/simple-sums-noend.jsonl --de-iters 500 --de-pop 8 --seeds 42

benchmark-full:
	python3 benchmark.py datasets/simple-sums-noend.jsonl --de-iters 2000 --de-pop 12 --seeds 42,7

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
