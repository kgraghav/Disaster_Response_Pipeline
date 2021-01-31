import sys
import pandas as pd
import numpy as np
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import matplotlib.pyplot as plt
import json
import plotly

from flask import Flask
from flask import Markup,render_template, request, jsonify
from plotly.graph_objs import Bar
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

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return 1
        return 0

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
   
def load_data(database_filepath):
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    df = pd.read_sql_table(database_filepath,engine)
    df.dropna(inplace=True)
    #print(df.head())
    X = df['message']
    Y = df[df.columns[4:-1]]
    category_names=df.columns[4:-1]
    return df, X,Y,category_names


# analyze accuracy
def display_results(y_test, y_pred):
    results=pd.DataFrame(columns=y_test.columns,index=['confusion_mat','precision','recall','f-score'])
    i=0
    for column in y_test.columns:
        results.loc['confusion_mat',column] = confusion_matrix(y_test.loc[:,column], y_pred[:,i])
        results.loc['precision',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[0]
        results.loc['recall',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[1]
        results.loc['f-score',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[2]
        i=i+1
    #fig=plt.figure(figsize=(12,8))
    #plt.barh(y_test.columns,results.loc['f-score',:])
    mean_accuracy_score=results.loc['f-score',:].mean()
    std_accuracy_score=results.loc['f-score',:].std()
    accuracy_sorted=list(results.loc['f-score',:].sort_values(ascending=True))
    lo_accuracy_labels=[]
    for sort_acc in accuracy_sorted[0:7]:
        for col in results.columns:
            if results.loc['f-score',col]==sort_acc and col not in lo_accuracy_labels:
                lo_accuracy_labels.append(col)
    return results,mean_accuracy_score,lo_accuracy_labels

# Test Model
def evaluate_model(model, X_test, Y_test):
    y_pred = model.predict(X_test)
    return y_pred

    
# load data 
database_filepath, model_filepath = sys.argv[1:]
print('Loading data...\n    DATABASE: {}'.format(database_filepath))
df,X, Y, category_names = load_data(database_filepath)
print(df.head())

# Train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.8)
X_train_model, X_test_model, Y_train_model, Y_test_model = train_test_split(X, Y, test_size=0.2,random_state=42)

# load model
model = joblib.load(model_filepath)

# Evaluate model
print('Evaluating model...')
y_pred=evaluate_model(model, X_test, Y_test)


# Display results
results,mean_accuracy_score,lo_accuracy_labels=display_results(Y_test, y_pred)
print('Mean Accuracy: {}'.format(mean_accuracy_score))
print('Lo Accuracy:{}'.format(lo_accuracy_labels))
print(results)


@app.route('/')
@app.route('/index')

# index webpage displays cool visuals and receives user input text for model
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_counts=[]
    train_category_counts=[]
    for i in range(len(Y.columns)):
        category_counts.append(Y.iloc[:,i].sum())
    for i in range(len(Y_train_model.columns)):
        train_category_counts.append(Y_train_model.iloc[:,i].sum())
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
                    'title': "Genre",
                    'tickangle':-45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=Y.columns,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Values in Entire "Y" Set per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':-45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=Y_train_model.columns,
                    y=train_category_counts
                )
            ],

            'layout': {
                'title': 'Values in Training Set per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':-45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=Y_test.columns,
                    y=results.loc['f-score',:],
                    orientation='v'
                )
            ],

            'layout': {
                'title': "Accuracy of Random Forest Grid Search CV Classifier on Test Data, Mean:{}".format(mean_accuracy_score),
                'yaxis': {
                    'title': "F-Score"
                },
                'xaxis': {
                    'title': 'Message category',
                    'tickangle':-45
                }
            }
        },
        {
            'data': [
                Bar(
                    x=train_category_counts,
                    y=results.loc['f-score',:]
                )
            ],

            'layout': {
                'title': 'Accuracy vs. Category counts',
                'yaxis': {
                    'title': "Accuracy"
                },
                'xaxis': {
                    'title': "Category",
                    'tickangle':-45
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


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
