ROOT_DIR=.

UV=uv
RUFF=$(UV) run ruff

install:
	$(UV) sync

update:
	$(UV) lock --upgrade
	$(UV) sync

lint:
	$(RUFF) check $(ROOT_DIR)

format:
	$(RUFF) check $(ROOT_DIR) --fix --ignore E731
	$(RUFF) format $(ROOT_DIR)

style: lint format

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

all: install style
	@echo "🎯 All tasks completed successfully."