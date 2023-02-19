# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 18:56:09 2023

@author: Datta K
"""
import json
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Heatmap
import plotly
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Data for second graph: Calculate proportions of different categories
    category_proportions = df.iloc[:, 4:].sum() / df.shape[0]
    xprop = category_proportions.index
    yprop = category_proportions.values

    # Data for third graph: Plot heatmap of message correlations
    correlations = df.iloc[:, 4:].corr()
    xcorr = correlations.columns
    ycorr = correlations.columns
    zcorr = correlations.values

    # create visuals
    graphs = [{
        'data': [Bar(x=genre_names, y=genre_counts)],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    }, {
        'data': [Bar(x=xprop, y=yprop)],
        'layout': {
            'title': 'Proportions of Message Categories',
            'yaxis': {
                'title': "Proportion"
            },
            'xaxis': {
                'title': "Category"
            },
            'margin': dict(l=150, r=50, t=50, b=150)
        }
    }, {
        'data': [
            Heatmap(x=xcorr,
                    y=ycorr,
                    z=zcorr,
                    colorscale='YlGnBu',
                    colorbar=dict(title='Correlation'))
        ],
        'layout': {
            'title': 'Message Categories Correlations Heatmap',
            'yaxis': {
                'title': "Category"
            },
            'xaxis': {
                'title': "Category"
            },
            'height': 800,
            'margin': dict(l=150, r=50, t=100, b=100)
        }
    }]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Save user input in query and use model to predict classification for query.
    Render go.html file with query and classification results.
    """
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template('go.html',
                           query=query,
                           classification_result=classification_results)


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
