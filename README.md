TimeBaseUla
==========================

<p align="center">
    <a href="https://dribia.github.io/timebaseula">
    <picture style="display: block; margin-left: auto; margin-right: auto; width: 40%;">
            <source
                media="(prefers-color-scheme: dark)"
                srcset="docs/img/logo_dribia_blanc_cropped.png"
            >
            <source
                media="(prefers-color-scheme: light)"
                srcset="docs/img/logo_dribia_blau_cropped.png"
            >
            <img
                alt="timebaseula"
                src="docs/img/logo_dribia_blau_cropped.png"
            >
        </picture>
    </a>
</p>

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| CI/CD   | [![Tests](https://github.com/dribia/timebaseula/actions/workflows/test.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/test.yml) [![Coverage Status](https://img.shields.io/codecov/c/github/dribia/timebaseula)](https://codecov.io/gh/dribia/timebaseula) [![Tests](https://github.com/dribia/timebaseula/actions/workflows/lint.yml/badge.svg)](https://github.com/dribia/timebaseula/actions/workflows/lint.yml) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| Package | [![PyPI](https://img.shields.io/pypi/v/timebaseula)](https://pypi.org/project/timebaseula/) ![PyPI - Downloads](https://img.shields.io/pypi/dm/timebaseula?color=blue&logo=pypi&logoColor=gold) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/timebaseula?logo=python&logoColor=gold) [![GitHub](https://img.shields.io/github/license/dribia/timebaseula?color=blue)](LICENSE)                                                                                                                                                                                                                                                                                                         |
---
**Documentation**: <a href="https://dribia.github.io/timebaseula" target="_blank">https://dribia.github.io/timebaseula</a>

**Source Code**: <a href="https://github.com/dribia/timebaseula" target="_blank">https://github.com/dribia/timebaseula</a>

---

## Installation

## Usage

### Examples

## Contributing
[uv](https://docs.astral.sh/uv/) is the best way to interact with this library, to install it,
follow the official [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

With `uv` installed, one can install the library dependencies with:

```shell
uv sync
```

Then, to run the library unit tests:

```shell
make test-unit
```

To run the linters (`ruff` and `mypy`):

```shell
make lint
```

To apply all code formatting:

```shell
make format
```

## License

timebaseula is distributed under the terms of
the [MIT](https://opensource.org/license/mit) license.
Check the [LICENSE](LICENSE) file for further details.
