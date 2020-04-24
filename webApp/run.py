import json
import plotly
import pandas as pd
import joblib

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# load classification report
with open("../models/classifier_report.json","r") as f:
    reports = json.load(f)

# load top unigrams
with open("../models/classifier_unigrams.json","r") as f:
    top_unigrams = json.load(f)

app = Flask(__name__)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # render web page with plotly graphs
    return render_template('master.html')

# web page to display information about dataset and classification reports
@app.route('/info')
def info():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # category_names = df.columns[4:]
    category_counts = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False).rename_axis('category')
    category_names = list(category_counts.index)

    report_category = []
    report_values = {'support': [],
                    'recall':   [],
                    'precision':[],
                    'f1-score': [],
                    }

    reports_ordered = []
    table = []
    for category in category_names:
        for cat, rep in reports:
            if cat == category:
                reports_ordered.append([cat, rep])

    for ix,(category, report) in enumerate(reports_ordered):
        support     = report['1']['support']
        recall      = report['1']['recall']
        precision   = report['1']['precision']
        f1_score    = report['1']['f1-score']
        report_category.append(category)
        report_values['support'].append(support)
        report_values['recall'].append(recall)
        report_values['precision'].append(precision)
        report_values['f1-score'].append(f1_score)
        table.append({'ix': ix,
                    'category': category.replace('_',' '),
                    'support': support,
                    'recall': f"{recall:.2f}",
                    'precision': f"{precision:.2f}",
                    'f1_score': f"{f1_score:.2f}",
                    'unigrams': ', '.join(top_unigrams[category])})

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {
            'data': [
                Bar(
                    x=category_counts,
                    y=[val.replace('_',' ') for val in category_names],
                    orientation='h'
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "",
                    'dtick': 1
                },
                'xaxis': {
                    'title': "Counts"
                },
                'width': 500,
                'height': 600,
                'margin': {
                    'l':150,
                    'r':10,
                    'b':50,
                    't':25,
                    'pad':4
                },
            }
        },

        {
            'data': [
                Bar(
                    x=[val.replace('_',' ') for val in report_category],
                    y=report_values['recall']
                )
            ],

            'layout': {
                'title': 'Recall',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                },
                'margin': {
                    'l':50,
                    'r':10,
                    'b':150,
                    't':25,
                    'pad':4
                },
            }
        },

        {
            'data': [
                Bar(
                    x=[val.replace('_',' ') for val in report_category],
                    y=report_values['precision']
                )
            ],

            'layout': {
                'title': 'Precision',
                'yaxis': {
                    'title': ""
                },
                'xaxis': {
                    'title': ""
                },
                'margin': {
                    'l':50,
                    'r':10,
                    'b':150,
                    't':25,
                    'pad':4
                },
            }
        },

        {
            'data': [
                Bar(
                    x=report_category,
                    y=report_values['f1-score']
                )
            ],

            'layout': {
                'title': 'Classification Report',
                'yaxis': {
                    'title': "F1-score"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    tableJSON = json.dumps(table)

    # render web page with plotly graphs
    return render_template('info.html',
    ids=ids, 
    graphJSON=graphJSON,
    tableJSON=tableJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
