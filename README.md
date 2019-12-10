# Personal webpage

Build by [Nikola](https://getnikola.com).

## Develop

1. Put `ipynb` file under `posts/`;
1. Add the following info into `metadata` in the notebook file ([ref](https://getnikola.com/handbook.html#jupyter-notebook-metadata)):

  "nikola": {
   "category": "",
   "date": "2018-11-04 19:44:01 UTC-08:00",
   "description": "",
   "link": "",
   "slug": "tensoflow-profiling",
   "tags": "",
   "title": "TensorFlow Profiling",
   "type": "text"
  }

1. Add `"<!-- TEASER_END -->\n"` to show something in the index page.
1. `nikola build` to build and test
1. `nikola github_deploy` to commit and push to github.io page


## Google Colab access

Most documents are developed in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jiayiliu/jiayiliu.github.io/blob/src/pages/index.ipynb)