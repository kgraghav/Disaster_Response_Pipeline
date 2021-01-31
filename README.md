# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root (workspace) directory in the IDE, to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's (workspace) directory to run your web app.
    `python run.py data/DisasterResponse.db models/classifier.pkl`
    
3. Run the following in the workspace directory in another terminal to obtain the access keys to get the authentication information
    (space-id and space-domain)

4. Open a web browser and type https://SPACEID-3001.SPACEDOMAIN with space id and space domain obtained from step 3 to access the dashboard



### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

The code was developed using <b>Python version 3.6.11.</b> Libraries used, include: <br>
1. pip - to install and manage packages from the [Python Package Index](https://pypi.org/)
2. numpy - [NumPy](https://numpy.org/) for array data handling
3. pandas - [Pandas](https://pandas.pydata.org/) for database and dataframe handling
4. sklearn - [Scikit-learn](https://scikit-learn.org/stable/)  machine learning and analysis module 
5. matplotlib - [Matplotlib](https://matplotlib.org/) visualization and plotting library
6. nltk - [NLTK](https://www.nltk.org/) Natural Language Toolkit

## Project Motivation<a name="motivation"></a>

For this project, I was interested in using machine learning on disaster relief data <br>
to be able to identify similar high priority messages in future, which could help allocate relief resources more efficiently

## File Descriptions <a name="files"></a>

The folder structure: <br>

    -workspace
        - template
        | |- master.html  # main page of web app
        | |- go.html  # classification result page of web app

        - data
        |- disaster_categories.csv  # categories data to process 
        |- disaster_messages.csv  # messages data to process
        |- process_data.py # python ETL Pipeline
        |- DisasterResponse.db.db   # database to save clean data to

        - models
        |- train_classifier.py
        |- classifier.pkl  # saved model 

        - run.py  # Flask file that runs app
    
 - ETL Pipeline Preparation.ipynb: ETL Pipeline notebook
 - ML Pipeline Preparation.ipynb: ML Pipelie notebook
 - Udacity_DRP_Report.pdf: Detailed report of the ETL and ML methods

## Results<a name="results"></a>

Please refer Udacity_DRP_Report.pdf for explanation of the project including analysis, results and conclusions

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

 Project done as part of Udacity's Data Scientist Nanodegree

