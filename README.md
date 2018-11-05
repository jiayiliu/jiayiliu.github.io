# Personal webpage

Build by [Nikola](https://getnikola.com).

## Develop

1. Put `ipynb` file under `posts/`;
2. Add the following info into `metadata` in the notebook file ([ref](https://getnikola.com/handbook.html#jupyter-notebook-metadata)):
    
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
3. `nikola build` to build and test
4. `nikola github_deploy` to commit and push to github.io page