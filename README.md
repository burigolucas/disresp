# Disaster Response Message Classifier

This project presents a classification of disaster response messages into a pre-defined set of categories. The data set is based on the [Multilingual Disaster Response Messages](https://appen.com/datasets/combined-disaster-response-data/) by Figure Eight.

### ETL and ML pipelines

The processing of the data is implemented in `data/process_data.py` and the messages classification in `modes/train_classifier.py`. 

The required data set of messages and annotations is available in `data/disaster_categories.csv` and `disaster_messages.csv`. The ETL script, `process_data.py`, takes the file paths of the two datasets and the path where to story a cleaned database as a SQLite database.
The script merges the messages and categories datasets, splitting the categories column into separate columns for each category, converting values to binary. Categories without any entries are droped. Besides, the category 'related' is also dropped as it is redundant with the categories 'aid related' and 'wheather related'.

The machine learning script, `train_classifier.py` is used to apply a multi-label classification algorithm. A grid search is applied to select the best classifier and tune the hyperparameters of the model and the text-preprocessing. For the text processing, a custom tokenize function using nltk is used to case normalize, lemmatize, and tokenize text.

There is a large imbalance of the labels in the data set. To best account for this, the split of the train/test set was performed using stratification and the metric f1 (weighted) was used for training.

### Web application

A flask-based web appliation deploying the model is implemented in `webApp/run.py`. The main page allows for the user to enter a message for classification. A info page is provided to display visualizations and a summary of the data set and the results of the training of the model

### Instructions:

0. (Recommended) Install requirements in a virtual environment
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

1. Run ETL pipeline in the project's root directory:

    ```
    python3 data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
    ```

2. Run ML pipeline that trains classifier and saves model:
  
    ```
    python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    ```

3. Deploy the web app:
    ```
    cd webApp
    python3 run.py
    ```

4. The web app can be accessed at http://0.0.0.0:3001/

### Requirements:

The app is developed in python3. The list below contains the python packages used. The list is provied in the `requirements.txt` file for easy installation with a package manager.
```
click==7.1.1
Flask==1.1.2
itsdangerous==1.1.0
Jinja2==2.11.2
joblib==0.14.1
MarkupSafe==1.1.1
nltk==3.5
numpy==1.18.3
pandas==1.0.3
plotly==4.6.0
python-dateutil==2.8.1
pytz==2019.3
regex==2020.4.4
retrying==1.3.3
scikit-learn==0.22.2.post1
scikit-multilearn==0.2.0
scipy==1.4.1
six==1.14.0
sklearn==0.0
SQLAlchemy==1.3.16
tqdm==4.45.0
Werkzeug==1.0.1
```