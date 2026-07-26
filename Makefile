# -----------------------------------------------------------------------------------------------------------
# lclaude Makefile
# -----------------------------------------------------------------------------------------------------------
# Usage:
#   make <target>
#   make run ARGS="--model ornith:35b"
#   make bump-version 0.4.0
# -----------------------------------------------------------------------------------------------------------

SHELL := /bin/bash

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
LOGGER := source "$(MAKEFILE_DIR)scripts/log.bash" &&
PYTHON ?= python3
DPRINT_CONFIG ?= $(HOME)/.config/dprint/dprint.json
RESET := \033[0m
DIM := \033[2m

ifneq ($(filter bump-version,$(MAKECMDGOALS)),)
BUMP_VERSION := $(word 2,$(MAKECMDGOALS))
ifeq ($(strip $(BUMP_VERSION)),)
$(error Usage: make bump-version <version>)
endif
ifneq ($(words $(MAKECMDGOALS)),2)
$(error Usage: make bump-version <version>)
endif
.PHONY: $(BUMP_VERSION)
$(BUMP_VERSION):
	@:
endif

.DEFAULT_GOAL := help

.PHONY: help bump-version format run test test-smoke test-unit

help: ## Show this help message
	@$(LOGGER) log_separator
	@$(LOGGER) log_banner
	@$(LOGGER) log_info "Available make targets:"
	@echo ""
	@(grep -E '^[[:space:]]*help:.*## .*$$' $(MAKEFILE_LIST) 2>/dev/null; grep -E '^[[:space:]]*[a-zA-Z0-9][a-zA-Z0-9_ -]*:.*## .*$$' $(MAKEFILE_LIST) | grep -v '^[[:space:]]*help:.*##' | sort) | \
			awk -F' ## ' '{ n = index($$1, ":"); target = substr($$1, 1, n-1); gsub(/^[ \t]+|[ \t]+$$/, "", target); desc = $$2; gsub(/^[ \t]+|[ \t]+$$/, "", desc); printf " %-22s$(RESET) $(DIM)- %s$(RESET)\n", target, desc }'
	@echo ""

bump-version: ## Set VERSION and sync Python script versions
	@$(LOGGER) log_target "Updating release version"
	@bash "$(MAKEFILE_DIR)scripts/bump-version.sh" "$(BUMP_VERSION)"
	@$(LOGGER) log_success "Version synchronized to v$(BUMP_VERSION)"

format: ## Format documentation with dprint
	@$(LOGGER) log_target "Formatting documentation"
	@$(LOGGER) log_run_dim dprint fmt --config "$(DPRINT_CONFIG)" README.md AGENTS.md
	@$(LOGGER) log_success "Formatting complete"

run: ## Run lclaude (pass CLI arguments with ARGS="...")
	@$(LOGGER) log_target "Running lclaude"
	@$(PYTHON) "$(MAKEFILE_DIR)lclaude.py" $(ARGS)

test: test-smoke test-unit ## Run all offline tests

test-smoke: ## Run syntax, version, and help smoke checks
	@$(LOGGER) log_target "Running Smoke Tests"
	@$(LOGGER) log_run_dim $(PYTHON) -c 'import ast; [ast.parse(open(path).read()) for path in ("lclaude.py", "lclaude-bench.py")]; print("Syntax OK")'
	@$(LOGGER) log_run_dim $(PYTHON) -c 'from pathlib import Path; version = Path("VERSION").read_text().strip(); paths = ("lclaude.py", "lclaude-bench.py"); assert all(f"__version__ = \"{version}\"" in Path(path).read_text() for path in paths); print(f"Versions synchronized: v{version}")'
	@$(LOGGER) log_run_dim bash -c '$(PYTHON) lclaude.py --help >/dev/null'
	@$(LOGGER) log_run_dim bash -c '$(PYTHON) lclaude.py --backend managed --help >/dev/null'
	@$(LOGGER) log_success "Smoke tests passed"
	@echo ""

test-unit: ## Run offline behavioral unittest suite
	@$(LOGGER) log_info "Running behavioral tests"
	@$(LOGGER) log_run_info $(PYTHON) -m tests.run
	@$(LOGGER) log_success "Behavioral tests passed"
	@echo ""