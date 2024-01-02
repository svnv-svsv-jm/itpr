# Information theoretic-based privacy risk evaluation for data anonymization

Reproducing results of the [ITPR paper](https://www.semanticscholar.org/reader/6c2a30bd365c698f7913bd87922410c39e06c8fc).

## Installation

To install the package `itpr`, git clone this repository, `cd` into it, and from its root directory, run (in a virtual environment):

```bash
pip install --upgrade pip poetry
poetry install
```

or

```bash
make install
```

## Development

Source code is located in `src/`: [here](./src/iptr/__init__.py). You can add code as you wish.

To test, run the following command:

```bash
make tests
```

## Usage

See notebook: [here](./notebooks/ITPR.ipynb).
