.PHONY: chat run train hello-world test-hello-world simple-sums simple-sums-mj simple-sums-stream simple-sums-noend simple-sums-me hello-world-me test-simple-sums micro-bench micro-identity micro-not micro-and micro-next micro-lower prove-identity-concept prove-identity-concept-end test-identity test-identity-end list-models help

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

micro-bench:
	python3 benchmark.py datasets/identity-bits-noend.jsonl --base dense --max 6 --de-iters 300 --de-pop 8 --seeds 42
	python3 benchmark.py datasets/not-bits-noend.jsonl --base dense --max 6 --de-iters 300 --de-pop 8 --seeds 42
	python3 benchmark.py datasets/and-bits-noend.jsonl --base dense --max 4 --de-iters 300 --de-pop 8 --seeds 42
	python3 benchmark.py datasets/next-digit-noend.jsonl --base dense --max 5 --de-iters 300 --de-pop 8 --seeds 42
	python3 benchmark.py datasets/lowercase-abc-noend.jsonl --base dense --max 5 --de-iters 300 --de-pop 8 --seeds 42

micro-identity:
	python3 trainer.py datasets/identity-bits-noend.jsonl --base dense --output identity-bits --multijoint

micro-not:
	python3 trainer.py datasets/not-bits-noend.jsonl --base dense --output not-bits --multijoint

micro-and:
	python3 trainer.py datasets/and-bits-noend.jsonl --base dense --output and-bits --multijoint

micro-next:
	python3 trainer.py datasets/next-digit-noend.jsonl --base dense --output next-digit --multijoint

micro-lower:
	python3 trainer.py datasets/lowercase-abc-noend.jsonl --base dense --output lowercase-abc --multijoint

prove-identity-concept:
	python3 experiments/prove_identity_concept.py

prove-identity-concept-end:
	python3 experiments/prove_identity_concept.py --model identity-concept-end --train datasets/identity-concept-end-train.jsonl --eval datasets/identity-concept-end-eval.jsonl

test-identity:
	python3 sinmachine.py --chat --model identity-concept

test-identity-end:
	python3 sinmachine.py --chat --model identity-concept-end

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
	@echo "  make micro-bench                     benchmark 5 tiny symbolic datasets"
	@echo "  make micro-identity                  train identity-bits model"
	@echo "  make micro-not                       train not-bits model"
	@echo "  make micro-and                       train and-bits model"
	@echo "  make micro-next                      train next-digit model"
	@echo "  make micro-lower                     train lowercase-abc model"
	@echo "  make prove-identity-concept          prove tiny identity abstraction"
	@echo "  make prove-identity-concept-end      prove identity abstraction with END"
	@echo "  make test-identity                   chat with identity-concept model"
	@echo "  make test-identity-end               chat with identity-concept-end model"
	@echo "  make list-models                     show available models"
	@echo ""
