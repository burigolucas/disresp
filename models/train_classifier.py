import sys
import pandas as pd
import numpy as np
import json

from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from skmultilearn.model_selection import iterative_train_test_split

from sklearn.feature_selection import chi2

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

def load_data(database_filepath, min_entries=1):
    '''
    INPUT:
    database_filepath   - (str) filepath of the database
    min_entries         - (int) minimum of entries to consider the category in the classification

    OUTPUT:
    X                   - (ndarray) features
    Y                   - (ndarray) labels
    category_names      - (list) name of categories
    '''
    # load database
    engine = create_engine('sqlite:///{:}'.format(database_filepath))  
    df = pd.read_sql_table('messages', engine)  

    # split features/labels
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
   
    # remove categories with nb of entries below min_entries
    if min_entries > 0:
        labels_value_counts = Y.sum(axis=0).sort_values().rename_axis('labels')
        labels_to_remove = labels_value_counts[labels_value_counts < min_entries].index
        Y.drop(labels_to_remove,axis=1,inplace=True)

    print("Nb of catetories to classify: {:}".format(Y.shape[1]))

    # extract label names
    category_names = Y.columns.tolist()

    # convert to ndarray
    X = X.values
    Y = Y.values
    
    return X, Y, category_names


def tokenize(text):
    '''
    INPUT:
    text            - (str) raw text 

    OUTPUT:
    clean_tokens    - (list) list of tokens

    Description:
    Tokenize the raw text using word_tokenize and WordNetLemmatizer from nltk
    removing stop words and cleaning url's
    '''
    # regex to clean up url's
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # get list of all urls using regex
    detected_urls = re.findall(url_regex,text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"url")
    
    # extra strings to be removed
    for string in ['http','bit.ly']:
        text = text.replace(string,'')
    
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text) 

    tokens = word_tokenize(text)

    # remove stop words
    tokens = [word for word in tokens if word not in stop_words]

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    '''
    INPUT:
    None 

    OUTPUT:
    model    - (GridSearchCV) model
    
    Description:
    The model is built using a pipeline with transformers CountVectorizer, TfidfTransformer
    and a multi-output random forest classifier 
    '''
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize,
                                max_features=None,
                                use_idf=False,
                                ngram_range=(1, 2))),
        ('clf', MultiOutputClassifier(LinearSVC(C=5)))
    ])

    # param_grid = {
    #     'vect__max_features': (None, 5000, 10000),
    #     'vect__ngram_range': ((1, 1), (1, 2)),
    #     'vect__use_idf': (True, False),
    # }

    param_grid = [
        {'clf' : [MultiOutputClassifier(LinearSVC())],
        'clf__estimator__C' : [0.1,0.5,1.,3.,5.,10],
        },
        {'clf' : [MultiOutputClassifier(RandomForestClassifier(n_estimators=100))],
        'clf__estimator__n_estimators' : [50, 100, 200],
        }
    ]

    model =  GridSearchCV(pipeline,
                          param_grid=param_grid,
                          verbose=1,
                          n_jobs=2,
                          scoring='f1_weighted',
                          cv=5)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model               - (GridSearchCV) model
    X                   - (ndarray) features
    Y                   - (ndarray) labels
    category_names      - (list) name of categories

    OUTPUT:
    classification_report - (list) report of classification for each category   

    Description:
    Evaluate the model accuracy, precision, and recall for each category
    '''
    # predict on the test data
    Y_pred = model.predict(X_test)

    classification_reports = []

    # evaluate model on each category
    for colIx, col in enumerate(category_names):
        print(col)
        print(classification_report(Y_test[:,colIx], Y_pred[:,colIx],output_dict=False))
        report = classification_report(Y_test[:,colIx], Y_pred[:,colIx],output_dict=True)
        classification_reports.append([col,report])

    accuracy = model.score(X_test,Y_test)
    f1_micro = f1_score(y_true=Y_test,y_pred=Y_pred,average="micro")
    f1_macro = f1_score(y_true=Y_test,y_pred=Y_pred,average="macro")
    f1_weighted = f1_score(y_true=Y_test,y_pred=Y_pred,average="weighted")

    print(f"Accuracy: {accuracy:.3f}\nF1 (micro): {f1_micro:.3f}\nF1 (macro): {f1_macro:.3f}\nF1 (weighted): {f1_weighted:.3f}")

    return classification_reports

def save_model(model, model_filepath):
    '''
    INPUT:
    model               - (GridSearchCV) model
    model_filepath      - (str) filepath where to save the model

    OUTPUT:
    None    

    Description:
    Dump the model to a binary file for persistency
    '''
    pickle.dump(model, open(model_filepath, 'wb'))   


def obtain_top_unigrams(X, Y, category_names, N=10):
    '''
    INPUT:
    X                   - (ndarray) features
    Y                   - (ndarray) labels
    category_names      - (list) name of categories

    OUTPUT:
    top_unigrams  - (list) list of top-N unigrams per category

    Description:
    Obtain the top unigrams per category by computing the correlation of features
    '''
    tfidf = TfidfVectorizer(tokenizer=tokenize,
                                max_features=None,
                                use_idf=False,
                                # ngram_range=(1, 2)
                                ngram_range=(1, 1)
                            )
    features = tfidf.fit_transform(X).toarray()

    top_unigrams = {}
    top_bigrams = {}
    for ix,category in enumerate(category_names):
        features_chi2 = chi2(features, Y[:,ix])
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        # bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        top_unigrams[category] = unigrams[-N:]
        # top_bigrams[category] = bigrams[-N:]

    return top_unigrams


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
                
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)

        # save top-unigrams to a file
        top_unigrams = obtain_top_unigrams(X, Y, category_names)

        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)

        # for iterative_train_test_split, X must be a (size,1) array
        X_train, Y_train, X_test, Y_test = iterative_train_test_split(X.reshape(X.size,1), Y, test_size=0.10)
        # setting the shape back to a (size,) array
        X_train, X_test = X_train.ravel(), X_test.ravel()

        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)

        print('Best model...')
        print(model.best_params_)
        print(model.cv_results_)

        print('Evaluating model...')
        report = evaluate_model(model, X_test, Y_test, category_names)

        with open("{:}_report.json".format(''.join(model_filepath.split('.')[:-1])), "w") as write_file:
            json.dump(report, write_file)

        with open("{:}_unigrams.json".format(''.join(model_filepath.split('.')[:-1])), "w") as write_file:
            json.dump(top_unigrams, write_file)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()