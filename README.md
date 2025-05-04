## authors

Annabel De Clerq, Romeo De Timmerman, Zaya Lips, Julie Van Wynsberge


## project structure

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Document folder.
│
├── results            <- PDF versions of notebooks and hyperparameter tuning CSV files.
│
├── models             <- Trained models.
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module.
│   │
│   ├── baseline.ipynb          <- Notebook for baseline model experiments.
│   ├── data_pipeline.py        <- Script for processing and preparing data.
│   ├── data_utils.py           <- Utility functions for data handling.
│   ├── exploration.ipynb       <- Notebook for exploratory data analysis.
│   ├── gradcam.ipynb           <- Notebook for Grad-CAM visualization experiments.
│   ├── gradcam.py              <- Script implementing Grad-CAM functionality.
│   ├── resnet.ipynb            <- Notebook for ResNet model experiments.
│   └── resnet.py               <- Script implementing the ResNet model.

(this project structure is based on the Cookiecutter Data Science template)
```
