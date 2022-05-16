Installation:

~~~
pip install -r requirements.txt
~~~
-------------------------------

Generate train and test corpus from raw data:
~~~
python ml_project/split.py -cf configs/config.yaml
~~~

Run training pipeline:
~~~
python ml_project/train.py -cf configs/config.yaml
~~~

Generate predictions:
~~~
python ml_project/predict.py -cf configs/config.yaml
~~~
-------------------------------
Tests:
~~~
py.test -v tests/tests.py
~~~

Project Organization
------------

    ├── README.md          					 <- The top-level README for developers about the project
    ├── configs            					 <- The top-level configs file
    ├── data
    │   ├── heart_cleveland_upload.csv.dvc   <- The original raw data dump
    │
    ├── ml_project
    │   ├── predict.py                       <- Script for generatining predictions
    │   ├── prepare_data.py                  <- Script to generate train/test corpus from raw data
    │   ├── train.py                         <- Script to train model
    │
    ├── model              					 <- Directory for trained model
    │
    ├── notebooks                            <- Jupyter notebooks
    |
    ├── requirements.txt                     <- The requirements file

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
=======
