.DEFAULT_GOAL := help
SHELL := /bin/bash

IMAGE       ?= bon-tts:0.1.0
CONFIG      ?= configs/librispeech_pc.yaml
GPU         ?= 0
OUTPUT_DIR  ?= ./outputs
CACHE_DIR   ?= ./cache
DATA_DIR    ?= ./data

# Verifiers scored once, then reused by every selection strategy.
VERIFIERS   ?= w2v2-base distil-sm distil-v3
EVALUATORS  ?= fwhisper-lgv3 w2v2-lv60 hubert-lg
# The cross-family pair behind the rank ensembles: wav2vec 2.0 + Whisper-distil.
ENSEMBLE    ?= w2v2-base distil-v3

SRC_MOUNTS = \
	-v $(CURDIR)/bon_tts:/workspace/bon_tts \
	-v $(CURDIR)/scripts:/workspace/scripts \
	-v $(CURDIR)/configs:/workspace/configs \
	-v $(CURDIR)/tests:/workspace/tests

DOCKER_RUN = docker run --rm -it --gpus all --shm-size=8g \
	-v $(abspath $(OUTPUT_DIR)):/outputs \
	-v $(abspath $(CACHE_DIR)):/cache \
	-v $(abspath $(DATA_DIR)):/data:ro \
	$(SRC_MOUNTS) \
	-e BON_TTS_OUTPUT_DIR=/outputs \
	$(IMAGE)

# Tests and lint need neither a GPU nor the data volumes, and must stay runnable
# where there is no TTY (CI) and no nvidia runtime.
DOCKER_RUN_CPU = docker run --rm $(SRC_MOUNTS) -e BON_TTS_OUTPUT_DIR=/outputs $(IMAGE)

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk -F':.*?## ' '{printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build the Docker image
	docker build -f docker/Dockerfile -t $(IMAGE) .

.PHONY: shell
shell: ## Open a shell in the container
	$(DOCKER_RUN) bash

.PHONY: test
test: ## Run the unit tests (CPU only)
	$(DOCKER_RUN_CPU) python -m pytest tests -q

.PHONY: lint
lint: ## Lint with ruff
	$(DOCKER_RUN_CPU) ruff check bon_tts scripts tests

.PHONY: lock
lock: ## Freeze the container's resolved versions into requirements.lock.txt
	@# `pip list --format=freeze`, not `pip freeze`: the conda-based torch image
	@# reports its own packages as `name @ file:///home/conda/...`, paths that do
	@# not exist anywhere else, which makes the lock file uninstallable. Drop the
	@# conda toolchain and the editable bon-tts install — base-image plumbing and
	@# the project itself, neither of them dependencies.
	@{ \
	  echo "# Exact versions resolved inside $(IMAGE) — the environment the paper's"; \
	  echo "# numbers were produced in. Regenerate with 'make lock'."; \
	  echo "#"; \
	  echo "# Install torch/torchaudio first: the +cu124 builds below come from the"; \
	  echo "# pytorch/pytorch base image, not PyPI. See requirements.txt."; \
	  echo "# The conda/mamba toolchain of the base image is excluded."; \
	  docker run --rm $(IMAGE) pip list --format=freeze \
	    | grep -viE '^(bon-tts|conda[-_a-z]*|mamba|libmambapy|menuinst)=='; \
	} > requirements.lock.txt
	@echo "wrote requirements.lock.txt"

.PHONY: pool
pool: ## Synthesize the N=10 candidate pool (long-running, GPU)
	$(DOCKER_RUN) python scripts/synthesize_pool.py --config $(CONFIG) --gpu $(GPU)

.PHONY: score
score: ## Score the pool with every verifier (GPU)
	@for v in $(VERIFIERS); do \
		echo "=== scoring $$v ==="; \
		$(DOCKER_RUN) python scripts/score_pool.py --config $(CONFIG) --verifier $$v --gpu $(GPU) || exit 1; \
	done

.PHONY: select
select: ## Run every selection configuration in the paper (CPU, fast)
	@for n in 3 5 10; do \
		for v in $(VERIFIERS); do \
			$(DOCKER_RUN) python scripts/select_candidates.py --config $(CONFIG) \
				--strategy single --verifiers $$v --n $$n || exit 1; \
		done; \
		for s in rank_avg max_rank; do \
			$(DOCKER_RUN) python scripts/select_candidates.py --config $(CONFIG) \
				--strategy $$s --verifiers $(ENSEMBLE) --n $$n || exit 1; \
		done; \
	done

.PHONY: evaluate
evaluate: ## Evaluate the baseline and every selection run under every evaluator (GPU)
	@for e in $(EVALUATORS); do \
		$(DOCKER_RUN) python scripts/evaluate.py --config $(CONFIG) --baseline \
			--evaluator $$e --gpu $(GPU) || exit 1; \
	done
	@for d in $(OUTPUT_DIR)/select_*; do \
		[ -f "$$d/selection.json" ] || continue; \
		for e in $(EVALUATORS); do \
			$(DOCKER_RUN) python scripts/evaluate.py --config $(CONFIG) \
				--run-name $$(basename $$d) --evaluator $$e --gpu $(GPU) || exit 1; \
		done; \
	done

.PHONY: report
report: ## Print the cross-evaluator table with significance tests (CPU)
	$(DOCKER_RUN) bash -c 'python scripts/report.py --config $(CONFIG) \
		--runs $$(ls -d /outputs/select_* | xargs -n1 basename | tr "\n" " ")'

.PHONY: clean-pycache
clean-pycache: ## Remove __pycache__ directories
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
