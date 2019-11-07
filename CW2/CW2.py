## Clean the data and remove the outliers
def clean_house(pathName, fileName = "Manhattan12.csv"):
    ## Load the data  
    import pandas as pd
    import numpy as np
    import os

    try:
        ## Read the .csv file
        pathName = pathName
        fileName = fileName
        filePath = os.path.join(pathName, fileName)
        house_price = pd.read_csv(filePath)

        print house_price.describe()
        print house_price.dtypes
        print house_price.shape

        ## Drop unneeded columns
        drop_list = ['BOROUGH', 'EASE-MENT', 'APART\nMENT\nNUMBER']    
        house_price.drop(drop_list, axis = 1, inplace = True)

        house_price['NEIGHBORHOOD'] = pd.core.strings.str_strip(house_price['NEIGHBORHOOD'])
        house_price['BUILDING CLASS CATEGORY'] = pd.core.strings.str_strip(house_price['BUILDING CLASS CATEGORY'])
        house_price['TAX CLASS AT PRESENT'] = pd.core.strings.str_strip(house_price['TAX CLASS AT PRESENT'])
        house_price['BUILDING CLASS AT PRESENT'] = pd.core.strings.str_strip(house_price['BUILDING CLASS AT PRESENT'])
        
        ## Remove duplicate rows
        house_price.drop_duplicates(inplace = True) 

        ## Remove rows with missing values
        for col in house_price:
            if house_price[col].dtype == "object":
                house_price = house_price[house_price[col] != ""]

        ## Clean Columns
        house_price["SALE\nPRICE"] = house_price["SALE\nPRICE"].replace("[^0-9]","",regex=True)
        house_price["LAND SQUARE FEET"] = house_price["LAND SQUARE FEET"].replace("[^0-9]","",regex=True)
        house_price["GROSS SQUARE FEET"] = house_price["GROSS SQUARE FEET"].replace("[^0-9]","",regex=True)
        house_price["RESIDENTIAL UNITS"] = house_price["RESIDENTIAL UNITS"].replace("[^0-9]","",regex=True)
        house_price["TOTAL UNITS"] = house_price["TOTAL UNITS"].replace("[^0-9]","",regex=True)

        ## Convert to numeric
        house_price["SALE\nPRICE"] = pd.to_numeric(house_price["SALE\nPRICE"], errors="raise")
        house_price["LAND SQUARE FEET"] = pd.to_numeric(house_price["LAND SQUARE FEET"], errors="raise")
        house_price["GROSS SQUARE FEET"] = pd.to_numeric(house_price["GROSS SQUARE FEET"], errors="raise")
        house_price["RESIDENTIAL UNITS"] = pd.to_numeric(house_price["RESIDENTIAL UNITS"], errors="raise")
        house_price["TOTAL UNITS"] = pd.to_numeric(house_price["TOTAL UNITS"], errors="raise")    

        ## Convert to datetime
        house_price["SALE DATE"] = pd.to_datetime(house_price["SALE DATE"], errors="raise")

        print "------------After Cleaning-----------"
        print house_price.describe()

        ## Remove zero values
        house_price = house_price[house_price['SALE\nPRICE'] != 0]
        house_price = house_price[house_price['GROSS SQUARE FEET'] != 0]
        house_price = house_price[house_price['TOTAL UNITS'] != 0]
        house_price = house_price[house_price['YEAR BUILT'] != 0]

        print house_price.describe()

        house_price = removeOutliers(house_price)
        house_price['lnprice'] = np.log(house_price['SALE\nPRICE'])
        print house_price.describe()
        house_price.to_csv('./cleaned_houseprice_data.csv')
        return house_price
    except Exception as e:
        print e
        raise e
        

## Remove outliers from the data 
def removeOutliers(house_price):
    import pandas as pd
    import numpy as np
    # Create an outlier column to mark outliers
    house_price['outlier'] = np.zeros(house_price.shape[0])
    house_price['outlier'] += [x<50000 for x in house_price['SALE\nPRICE']]
    house_price['outlier'] += [x<1800 for x in house_price['YEAR BUILT']]
    house_price['outlier'] += [x<500 for x in house_price['GROSS SQUARE FEET']]
    house_price['outlier'] += [x>100 for x in house_price['TOTAL UNITS']]
    house_price = house_price[house_price.outlier == False] # filter for outliers
    house_price = house_price.drop('outlier', axis=1)

    return house_price

##Create a predictive model and visualise the predictions
def model(house_price):
    # Run this code interactively for a better understanding
    # Assume your clean data is in df
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import train_test_split
    from sklearn.feature_selection import RFE
    from sklearn.svm import SVR
    # Assume your clean data is in house_price
    df = normalize(house_price)
    # all columns of the data
    # feature_cols = df.columns.values.tolist()
    # feature_cols = df._get_numeric_data().columns.values.tolist()
    feature_cols = ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 'TAX CLASS AT TIME OF SALE','SALE\nPRICE','lnprice']
    print feature_cols
    # remove the outcome 'lnprice'
    feature_cols.remove('lnprice')
    feature_cols.remove('SALE\nPRICE')
    X0 = df[feature_cols]
    Y0 = df['lnprice']
    estimator = SVR(kernel="linear")
    # We desire a linear model with 5 variables
    # If we want 2 features, we change 5 to 2; etc.
    selector = RFE(estimator,4,step=1)
    selector = selector.fit(X0, Y0)
    selector.support_
    rank = selector.ranking_
    print "ranks :    ",rank
    # From the ranking you can select your predictors with rank 1
    # Model 1; let us select the following features as predictors:
    temp = np.array(feature_cols)
    # defining a mask
    msk = rank==1
    select_features = temp[msk].tolist()
    X = df[select_features]
    Y = df['lnprice']
    trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.4)

    lm = LinearRegression()
    lm.fit(trainX, trainY)
    print 'Weight Coefficients', lm.coef_
    print 'Y-axis intercept', lm.intercept_
    # the equation of the model.
    zip(select_features, lm.coef_)
    # predicting the outcomes from the test sample
    # the value of R^2.
    print 'R squared for the training data is: ', lm.score(trainX, trainY)
    # Plotting the regression fit of the traning data
    fig = plt.figure(figsize=(14, 4))
    fig.clf()
    min_pt = X.min()*lm.coef_[0] + lm.intercept_
    max_pt = X.max()*lm.coef_[0] + lm.intercept_
    plt.plot([X.min(), X.max()], [min_pt, max_pt])
    plt.plot(trainX, trainY, 'o')
    plt.show()
    #Predicting the Ys from the train data
    pred_trainY = lm.predict(trainX)
    #Plotting
    fig.clf()
    plt.plot(trainX, trainY, 'o', label="Observed")
    plt.plot(trainX, pred_trainY, 'o', label="Predicted")
    plt.plot([X.min(), X.max()], [min_pt, max_pt], label='fit')
    plt.legend(loc='Best')
    plt.show()
    #Predicting the Ys from the test data
    pred_testY = lm.predict(testX)
    # the value of R^2.
    print 'R squared for the test data is: ', lm.score(testX, testY)

    predicted_prices = lm.predict(df[select_features])
    plt.figure(figsize=(12, 8))
    plt.scatter(df['lnprice'], predicted_prices)
    plt.plot(range(2),range(2),color='darkorange')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Plot of predicted vs actual prices")
    plt.show()

## Normalise all data values to between 0 and 1
def normalize(df):
    import numpy as np
    # select all numeric columns of df except 'lnprice'
    allcols = df.columns.values.tolist()
    num_cols = []
    for col in allcols:
        if(df[col].dtype in [np.int64, np.int32, np.float64]):
            num_cols.append(col)
    # normalize the dataset using this transformation
    df_norm = (df[num_cols]-df[num_cols].min())/(df[num_cols].max()-df[num_cols].min())
    return df_norm

## Create pair-wise scatter plots
def house_pairs(plot_cols, df):
    import matplotlib.pyplot as plt
    from pandas.tools.plotting import scatter_matrix
    fig = plt.figure(1, figsize=(12, 12))
    fig.clf()
    ax = fig.gca()
    scatter_matrix(df[plot_cols], alpha=0.3,
            diagonal='kde', ax = ax)
    plt.show()
    return('Done')

## Create Boxplots of data
def house_boxplot(df, plot_cols, by):
    import matplotlib.pyplot as plt
    for col in plot_cols:
        fig = plt.figure(figsize=(9, 6))
        ax = fig.gca()
        df.boxplot(column = col, by = by, ax = ax)
        ax.set_title('Box plots of ' + col + ' by ' + by)
        ax.set_ylabel(col)
        plt.xticks(rotation = 90)
        plt.show()
    return by

def main():
    data = clean_house("")
    # house_boxplot(data, ['SALE\nPRICE'],'NEIGHBORHOOD')
    # cols = ['RESIDENTIAL UNITS','COMMERCIAL UNITS', 'TOTAL UNITS', 'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'SALE DATE', 'SALE\nPRICE']
    # house_pairs(cols, data)
    model(data)

if __name__ == '__main__':
    main()