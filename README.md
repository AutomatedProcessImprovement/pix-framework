# PIX Framework

![pix-framework](https://github.com/AutomatedProcessImprovement/pix-framework/actions/workflows/build.yaml/badge.svg)
![version](https://img.shields.io/github/v/tag/AutomatedProcessImprovement/pix-framework)

Framework for building process mining applications.

## Installation

The package requires **Python 3.9+**. You can install it from PyPI: 

```shell
pip install pix-framework
```

## Releases

You can browse compiled releases in the [Releases](https://github.com/AutomatedProcessImprovement/pix-framework/releases) section.

## Description

The framework is composed of the following packages:

- [pix_framework](./src/pix_framework/)
- [batch_processing_discovery](./src/batch_processing_discovery/)
- [prioritization_discovery](./src/prioritization_discovery/)
- [start_time_estimator](./src/start_time_estimator/)

## Development

### Testing

To run tests, use the following command:

```shell
pytest -m "not benchmark"
```

### Benchmarking

To run benchmarks, use the following command:

```shell
pytest --benchmark-only
```
