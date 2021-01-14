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


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}.db'.format(database_filepath))
    df = pd.read_sql_table("data/DisasterResponse.db",engine)
    df.dropna(inplace=True)
    print(df.head())
    X = df['message']
    Y = df[df.columns[4:-1]]
    category_names=df.columns[4:-1]
    return X,Y,category_names

def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = re.sub(r"[0-9]", "", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens=[word for word in tokens if word in list(wn.words())]
    tokens=[word for word,pos in pos_tag(tokens) if not pos=='UH']
    tokens=[word for word,pos in pos_tag(tokens) if not pos=='NNP']
    tokens=[word for word,pos in pos_tag(tokens) if not pos=='PRP']
    
    if len(tokens)==0:
        tokens=['none']

    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(tokenizer=tokenize)),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
    ])
    return pipeline

def display_results(y_test, y_pred):
    results=pd.DataFrame(columns=y_test.columns,index=['confusion_mat','precision','recall','f-score'])
    i=0
    for column in y_test.columns:
        results.loc['confusion_mat',column] = confusion_matrix(y_test.loc[:,column], y_pred[:,i])
        results.loc['precision',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[0]
        results.loc['recall',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[1]
        results.loc['f-score',column]=precision_recall_fscore_support(y_test.loc[:,column] ,y_pred[:,i],average='micro')[2]
        i=i+1
    fig=plt.figure(figsize=(12,8))
    plt.barh(y_test.columns,results.loc['f-score',:])
    mean_accuracy_score=results.loc['f-score',:].mean()
    std_accuracy_score=results.loc['f-score',:].std()
    accuracy_sorted=list(results.loc['f-score',:].sort_values(ascending=True))
    lo_accuracy_labels=[]
    for sort_acc in accuracy_sorted[0:7]:
        for col in results.columns:
            if results.loc['f-score',col]==sort_acc and col not in lo_accuracy_labels:
                lo_accuracy_labels.append(col)
    return results,mean_accuracy_score,lo_accuracy_labels    

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    results,mean_accuracy_score,lo_accuracy_labels=display_results(Y_test, y_pred)
    print('Mean Accuracy: {}'.format(mean_accuracy_score))
    print('Lo Accuracy:{}'.format(lo_accuracy_labels))
    print(results)
    return results


def save_model(model):
    outfile=open(model_filepath,'wb')
    pickle.dump( model,outfile  )
    outfile.close()


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
