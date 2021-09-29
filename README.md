[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
<h3 align="center">NLDA - Natural Language Data Augmentation </h3>
<p align="center">
Augment your Natural Language data easily.
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

NLDA aims to ease Natural Language Data Augmentation by enlarging your dataset and thus providing better Machine
Learning models for any task.


<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

```shell
git clone https://github.com/Sunwaee/nlda.git
```

2. Install requirements

```shell
pip install -r requirements.txt
```

<!-- USAGE EXAMPLES -->

## Usage

### Prerequisites

- .tsv file containing atleast 2 columns, with source and target text

### Example

Here is an example of how to use NLDA from `main.py`. This takes sentences contained in the `source_text` column of
the `example.tsv` dataset and will translate them to `target_languages`. It will use CUDA if available and CPU otherwise.

```python
from nlda import translate_dataset

# Dataset informations
path_to_dataset = 'example.tsv'
target_languages = ['en', 'fr']

# Augment dataset using translation with default arguments
translate_dataset(path_to_dataset, target_languages=target_languages)
```

> Note that other arguments can be passed, including `source_col`, `target_col`, `translate_targets` and `aug_data_path`.

Output should be as follows:

```shell
INFO:root:Currently running NLDA. 

Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.
INFO:root: Dataset stats are the following:
en    0.333333
it    0.333333
fr    0.333333
Name: language, dtype: float64

INFO:root: Translating your dataset to ['en', 'fr']. Please wait...
100%|██████████| 3/3 [00:35<00:00, 11.73s/it]

INFO:root: Augmented dataset has been saved to augmented_data.tsv. 
```

<!-- CONTRIBUTING -->

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->

## Contact

David NAISSE - [@LinkedIn](https://www.linkedin.com/in/davidnaisse/) - private.david.naisse@gmail.com



<!-- ACKNOWLEDGEMENTS -->

## Acknowledgements

* None

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/sunwaee/nlda.svg?style=for-the-badge

[contributors-url]: https://github.com/Sunwaee/nlda/graphs/contributors

[forks-shield]: https://img.shields.io/github/forks/sunwaee/nlda.svg?style=for-the-badge

[forks-url]: https://github.com/Sunwaee/nlda/network/members

[stars-shield]: https://img.shields.io/github/stars/sunwaee/nlda.svg?style=for-the-badge

[stars-url]: https://github.com/Sunwaee/nlda/stargazers

[issues-shield]: https://img.shields.io/github/issues/sunwaee/nlda.svg?style=for-the-badge

[issues-url]: https://github.com/Sunwaee/nlda/issues

[license-shield]: https://img.shields.io/github/license/sunwaee/nlda.svg?style=for-the-badge

[license-url]: https://github.com/Sunwaee/nlda/blob/master/LICENSE.txt

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555

[linkedin-url]: https://www.linkedin.com/in/davidnaisse/