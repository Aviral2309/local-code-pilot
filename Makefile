# AuraLSP Makefile
# ----------------
# All common operations in one place.
# Run `make help` to see available commands.

PYTHON      := python3
PIP         := pip3
SRC_DIR     := src
TEST_DIR    := tests
CONFIG_FILE := config/default.json

.PHONY: help install install-dev test test-cov lint server metrics docker docker-down clean bench

# ---------------------------------------------------------------------------
help:   ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	| sort \
	| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
install:  ## Install AuraLSP and all runtime dependencies
	$(PIP) install -e .
	@echo "✓ AuraLSP installed. Run 'make server' to start the LSP server."

install-dev:  ## Install including dev/test dependencies
	$(PIP) install -e ".[dev]"
	@echo "✓ Dev dependencies installed. Run 'make test' to run tests."

# ---------------------------------------------------------------------------
test:  ## Run all unit tests
	pytest $(TEST_DIR) -v --tb=short

test-cov:  ## Run tests with coverage report
	pytest $(TEST_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report at htmlcov/index.html"

test-fast:  ## Run tests skipping slow integration tests
	pytest $(TEST_DIR) -v --tb=short -m "not slow"

# ---------------------------------------------------------------------------
server:  ## Start the AuraLSP LSP server (stdio mode for editor integration)
	AURALSP_CONFIG=$(CONFIG_FILE) $(PYTHON) -m src.server

metrics:  ## Start the metrics API server on localhost:8765
	AURALSP_CONFIG=$(CONFIG_FILE) $(PYTHON) -m src.metrics_api
	@echo "Metrics API: http://localhost:8765"
	@echo "Swagger UI:  http://localhost:8765/docs"

# ---------------------------------------------------------------------------
docker:  ## Start Ollama + metrics API via docker-compose
	docker-compose up -d
	@echo "✓ Services started."
	@echo "  Ollama:  http://localhost:11434"
	@echo "  Metrics: http://localhost:8765"

docker-down:  ## Stop all docker-compose services
	docker-compose down

docker-logs:  ## Stream logs from all services
	docker-compose logs -f

# ---------------------------------------------------------------------------
ollama-pull:  ## Pull the Qwen2.5-Coder model (run once after installing Ollama)
	ollama pull qwen2.5-coder:1.5b
	@echo "✓ Model ready. Test with: curl http://localhost:11434/api/tags"

ollama-test:  ## Test Ollama connectivity
	curl -s http://localhost:11434/api/tags | python3 -m json.tool

# ---------------------------------------------------------------------------
bench:  ## Run the evaluation harness (Phase 3)
	@echo "Evaluation harness not yet implemented (Phase 3)."
	@echo "Run: python -m eval.eval_harness"

# ---------------------------------------------------------------------------
lint:  ## Check code style (requires ruff: pip install ruff)
	ruff check $(SRC_DIR) $(TEST_DIR) --select=E,W,F

clean:  ## Remove generated files and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage auralsp.log
	@echo "✓ Cleaned."
