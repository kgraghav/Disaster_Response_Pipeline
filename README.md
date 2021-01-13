# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/



### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code was run using <b>Python version 3.6.11.</b> Libraries used, include: <br>
1. pip - to install and manage packages from the [Python Package Index](https://pypi.org/)
2. numpy - [NumPy](https://numpy.org/) for array data handling
3. pandas - [Pandas](https://pandas.pydata.org/) for database and dataframe handling
6. sklearn - [Scikit-learn](https://scikit-learn.org/stable/)  machine learning and analysis module 
                Specifically K-means
7. matplotlib - [Matplotlib](https://matplotlib.org/) visualization and plotting library

## Project Motivation<a name="motivation"></a>

For this project, I was interested in using machine learning on emergency data <br>
to be able to identify similar high priority messages in future.

## File Descriptions <a name="files"></a>

The folder structure: <br>
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

## Results<a name="results"></a>

WIP

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

 WIP

