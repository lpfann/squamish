<h1 align="center">Welcome to squamish (WIP)</h1>
<p>
  <a href="#" target="_blank">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg" />
  </a>
</p>

> This project contains a novel feature selection algorithm which performs feature classification and selection using a Random Forest classifier (LightGBM) and Boruta.
> 
> It classifies each data feature into the three classes (1) strong relevant features (2) weakly relevant features (3) irrelevant features.
> 
> It also searches for related features in the group of weakly relevant features.
> As such it shows not only one possible optimal model but several at the same time.
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

## Author

👤 **Lukas Pfannschmidt**

* Website: https://lpfann.me
* Twitter: [@lpfann](https://twitter.com/lpfann)
* Github: [@lpfann](https://github.com/lpfann)


***
_This README was generated with ❤️ by [readme-md-generator](https://github.com/kefranabg/readme-md-generator)_