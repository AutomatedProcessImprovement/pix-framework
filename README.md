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

The `pix_framework.discovery` package besides root modules contains the following subpackages with additional information located in their README files:

- [batch_processing_discovery](src/pix_framework/discovery/batch_processing/)
- [prioritization_discovery](src/pix_framework/discovery/prioritization/)
- [case_attribute_discovery](src/pix_framework/discovery/case_attribute/)

The `pix_framework.enhancement` package besides root modules contains the following subpackages with additional information located in their README files:

- [start_time_estimator](src/pix_framework/enhancement/start_time_estimator/)

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
