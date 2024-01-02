help:
	@cat Makefile

.EXPORT_ALL_VARIABLES:

# create an .env file to override the default settings
-include .env
export $(shell sed 's/=.*//' .env)


# ----------------
# default settings
# ----------------
PYTHON_EXEC?=python -m
COV_FAIL_UNDER?=35


# -----------
# install project's dependencies
# -----------
install:
	$(PYTHON_EXEC) pip install --upgrade pip
	$(PYTHON_EXEC) pip install --upgrade poetry


# -----------
# testing
# -----------
pytest:
	$(PYTHON_EXEC) pytest -x --testmon --nbmake --overwrite "./notebooks"
	$(PYTHON_EXEC) mypy tests
	$(PYTHON_EXEC) pytest -x --testmon --pylint --cov-fail-under $(COV_FAIL_UNDER)

tests: pytest