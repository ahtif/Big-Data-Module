def read_income(fileName = "Adult_Census_Income.csv"):
    ## Load the data  
    import pandas as pd
    return pd.read_csv(fileName)

def clean_income(df):
    ## Remove rows with missing values
    df.dropna(axis = 0, inplace = True)
    ## Drop specific rows 
    ## Drop rows with ? in column 'workclass'
    for i in range(0, len(df)):
    	if (df['workclass'][i]==' ?'):
            df.drop(df.index[i])
    return df

## Plot categorical variables as bar plots
def income_barplot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    cols = df.columns.tolist()[:-1]
    for col in cols:
        if(df.ix[:, col].dtype not in [np.int64, np.int32, np.float64]):
            temp1 = df.ix[df['income'] == ' <=50K', col].value_counts()
            temp0 = df.ix[df['income'] == ' >50K', col].value_counts() 
            
            ylim = [0, max(max(temp1), max(temp0))]
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax0 = fig.add_subplot(1, 2, 2) 
            temp1.plot(kind = 'bar', ax = ax1, ylim = ylim)
            ax1.set_title('Values of ' + col + '\n for income <= 50K')
            temp0.plot(kind = 'bar', ax = ax0, ylim = ylim)
            ax0.set_title('Values of ' + col + '\n for income > 50K')
            plt.show()
    return('Done')            
            
            
## Plot categorical variables as box plots
def income_boxplot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    cols = df.columns.tolist()[:-1]
    for col in cols:
        if(df[col].dtype in [np.int64, np.int32, np.float64]):                  
            fig = plt.figure(figsize = (6,6))
            fig.clf()
            ax = fig.gca() 
            df.boxplot(column = [col], ax = ax, by = ['income'])  
            plt.show()        
    return('Done')  

## scaling/normalising data
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

## creating dummy variables from a given categorical
def dummy_var(df, var):
    import pandas as pd
    dum_var = pd.get_dummies(df[var], prefix=var)	
    col_names = df.columns.values.tolist() 
    df1 = df[col_names].join(dum_var)
    return df1
	
