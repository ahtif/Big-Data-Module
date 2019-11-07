# Scaling data
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

def id_outlier(df):
    ## Create a vector of 0 of length equal to the number of rows
    temp = [0] * df.shape[0]
    ## test each outlier condition and mark with a 1 as required
    for i, x in enumerate(df['engine-size']):
        if (x > 190): temp[i] = 1 
    for i, x in enumerate(df['curb-weight']):
        if (x > 3500): temp[i] = 1 
    for i, x in enumerate(df['city-mpg']):
        if (x > 40): temp[i] = 1      
    df['outlier'] = temp # append a column to the data frame
    return df


def clean_auto(pathName, fileName = "Automobile price data _Raw.csv"):
    ## Load the data  
    import pandas as pd
    import numpy as np
    import os

    ## Read the .csv file
    pathName = pathName
    fileName = fileName
    filePath = os.path.join(pathName, fileName)
    auto_price = pd.read_csv(filePath)

    ## Convert some columns to numeric values
    cols = ['price', 'bore', 'stroke', 
          'horsepower', 'peak-rpm']
    auto_price[cols] = auto_price[cols].convert_objects(convert_numeric = True)
    
    ## Drop unneeded columns
    drop_list = ['symboling', 'normalized-losses']    
    auto_price.drop(drop_list, axis = 1, inplace = True)

    
    ## Remove duplicate rows
    auto_price.drop_duplicates(inplace = True) 

    ## Remove rows with missing values
    auto_price.dropna(axis = 0, inplace = True)

    ## Compute the log of the auto price
    auto_price['lnprice'] = np.log(auto_price.price)

    ## Create a column with new levels for the number of cylinders
    auto_price['num-cylinders'] = ['four-or-less' if x in ['two', 'three', 'four'] else 
                                 ('five-six' if x in ['five', 'six'] else 
                                 'eight-twelve') for x in auto_price['num-of-cylinders']]
    
    ## Summary with outliers
    # print "With outliers\n", auto_price.describe(), "\n"

    ## Removing outliers
    auto_price = id_outlier(auto_price)  # mark outliers       
    auto_price = auto_price[auto_price.outlier == 0] # filter for outliers
    auto_price.drop('outlier', axis = 1, inplace = True)

    ## Summary without outliers
    # print "Without outliers\n", auto_price.describe() ,"\n"

    ###
    auto_price.to_csv('./cleaned_autoprice_data.csv')
    return auto_price

