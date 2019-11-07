import pandas as pd
import numpy as np
import exploreIncome as ex

def cleanCensus(df):
    ## Remove rows with missing values
    df.dropna(axis = 0, inplace = True)
    ## Drop specific rows 
    ## Drop rows with ? in column 'workclass'
    
    # for col in df:
    #     df = df[df[col] != " ?"]

    df = df[df["workclass"] != " ?"]
    return df

def normalise(df):
    normalised = ex.normalize(df)
    for col in normalised:
        df[col] = normalised[col]
    return df

def logistic_model(df):
    from sklearn import linear_model
    from sklearn.cross_validation import train_test_split
    clf = linear_model.LogisticRegression()
    # start with the following predictors
    f_cols = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    X = df[f_cols]
    Y = df['income']
    trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.4)
    clf.fit(trainX, trainY)
    clf.coef_
    clf.intercept_
    clf.score(trainX, trainY)
    from sklearn.cross_validation import cross_val_score
    train_scores = cross_val_score(linear_model.LogisticRegression(),
    trainX, trainY, scoring='accuracy', cv=8)
    print "Training Scores:"
    print train_scores
    print train_scores.mean()

    test_scores = cross_val_score(linear_model.LogisticRegression(),
    testX, testY, scoring='accuracy', cv=8)
    print "\n Test Scores"
    print test_scores
    print test_scores.mean()
    

def main():
    census = pd.read_csv("Adult_Census_Income.csv")
    census = cleanCensus(census)
    print census.describe()
    print census.dtypes
    print census.shape
    print census.head(5)
    
    # ex.income_barplot(census)
    # ex.income_boxplot(census)
    census = normalise(census)
    
    print census.dtypes
    logistic_model(census)

if __name__ == '__main__':
    main()