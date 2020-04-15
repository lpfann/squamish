<h1 align="center">Welcome to squamish </h1>
<p>
  <a href="#" target="_blank">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache_2.0-yellow.svg" />
  </a>
</p>

> This project contains a novel feature selection algorithm which performs feature classification and selection using a Random Forest classifier (LightGBM) and Boruta.
> 
> It classifies each data feature into the three classes (1) strong relevant features (2) weakly relevant features (3) irrelevant features.
> 
>A Publication detailing the methods used here is WIP.
>
> The name is a codename without meaning and chosen because of personal reasons (Beautiful British Columbia...)


## Install
We use [poetry](https://python-poetry.org/) as our dependency manager and packaging tool.

```sh
poetry install
```

## Run tests

```sh
poetry run pytest
```

# Cite

```bibtex
@misc{pfannschmidt2020sequential,
    title={Sequential Feature Classification in the Context of Redundancies},
    author={Lukas Pfannschmidt and Barbara Hammer},
    year={2020},
    eprint={2004.00658},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
Preprints can be found at https://pub.uni-bielefeld.de/record/2942271 or https://arxiv.org/abs/2004.00658.
Experiments of the papers are located [here](https://github.com/lpfann/squamish_experiments).
