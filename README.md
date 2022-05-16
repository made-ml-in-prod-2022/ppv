Installation:

~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
~~~
-------------------------------
All settings are set in configs/(config_train.yml and config_test.yml) files

Two modes of operation: training and predicting.

Run training pipeline:
~~~
python run.py train
~~~

Run predict pipeline:
~~~
python run.py predict -d path_to_data -s path_to_save_predicts
~~~
-------------------------------

Project Organization
------------

    ├── README.md          					 <- The top-level README for developers about the project
    ├── configs            					 <- Configs to train and predict pipelines
    ├── data
    │   ├── heart_cleveland_upload.csv.dvc   <- The original raw data
	│   ├── training                         <- Additional directory for saving data during training models 
    │
    ├── src
    │   ├── predict.py                       <- Script for generatining predictions
    │   ├── prepare_data.py                  <- Script to generate train/test corpus from raw data
    │   ├── train.py                         <- Script to train model
    │
    ├── model              					 <- Directory for trained models
    │
    ├── notebooks                            <- Jupyter notebooks
    |
    ├── requirements.txt                     <- The requirements file


-------------------------------
