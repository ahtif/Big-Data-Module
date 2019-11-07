def clean_credit(pathName, fileName = "credit_scores.csv"):
    ## Load the data  
    import pandas as pd
    import numpy as np
    import os

    try:
        ## Read the .csv file
        pathName = pathName
        fileName = fileName
        filePath = os.path.join(pathName, fileName)
        credit = pd.read_csv(filePath)

        print credit.describe()
        print credit.dtypes
        print credit.shape

        ## Drop unneeded columns
        drop_list = ['Telephone']    
        credit.drop(drop_list, axis = 1, inplace = True)

        ## Remove duplicate rows
        credit.drop_duplicates(inplace = True) 

        return credit
    except Exception as e:
        print e
        raise e

def plot_column(df, column):
    import numpy as np
    import matplotlib.pyplot as plt
    
    plt.hist(df[column])
    plt.title('Credit Status')
    plt.show()
    plt.clf()

## Plot categorical variables as bar plots
def credit_barplot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    cols = df.columns.tolist()[:-1]
    for col in cols:
        if(df.ix[:, col].dtype not in [np.int64, np.int32, np.float64]):
            temp1 = df.ix[df['CreditStatus'] == 0, col].value_counts()
            temp0 = df.ix[df['CreditStatus'] == 1, col].value_counts() 
            
            ylim = [0, max(max(temp1), max(temp0))]
            fig = plt.figure(figsize = (12,6))
            fig.clf()
            ax1 = fig.add_subplot(1, 2, 1)
            ax0 = fig.add_subplot(1, 2, 2) 
            temp1.plot(kind = 'bar', ax = ax1, ylim = ylim)
            ax1.set_title('Values of ' + col + '\n for credit status = 0')
            temp0.plot(kind = 'bar', ax = ax0, ylim = ylim)
            ax0.set_title('Values of ' + col + '\n for credit status = 1')
            plt.show()
    return('Done')            
            
            
## Plot categorical variables as box plots
def credit_boxplot(df):
    import numpy as np
    import matplotlib.pyplot as plt
    
    cols = df.columns.tolist()[:-1]
    for col in cols:
        if(df[col].dtype in [np.int64, np.int32, np.float64]):                  
            fig = plt.figure(figsize = (6,6))
            fig.clf()
            ax = fig.gca() 
            df.boxplot(column = [col], ax = ax, by = ['CreditStatus'])  
            plt.show()        
    return('Done')  

#Create dummy variables
def create_dummies(df):
    import pandas as pd
    ## create the dummies
    cat_vars=['CheckingAcctStat','CreditHistory','Purpose','Savings','Employment','SexAndStatus',
                'OtherDetorsGuarantors','Property','OtherInstalments','Housing','Job','ForeignWorker']
    for var in cat_vars:
        cat_list='var'+'_'+var
        cat_list = pd.get_dummies(df[var], prefix=var)
        df1 = df.join(cat_list)
        df = df1
    ## select the binary variables
    df_vars=df.columns.values.tolist()
    to_keep=[i for i in df_vars if i not in cat_vars]
    ## remove the original variables
    df = df[to_keep]
    return df

#Select the best features using sklearn 
def select_features(df):
    from sklearn import datasets
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    
    model = LogisticRegression()
    preds = df.columns.values.tolist()
    predictors = [i for i in preds if i not in ['CreditStatus']]
    X0 = df[predictors]
    Y0 = df['CreditStatus']
    # select 12 predictors
    selector = RFE(model,15,step=1)
    selector = selector.fit(X0, Y0)
    print(selector.support_)
    print "ranks: ", (selector.ranking_)
    rank = selector.ranking_
    temp = np.array(predictors)
    # defining a mask
    msk = rank==1
    s_features = temp[msk].tolist()
    return s_features

#Create a predictive logistical regression model and evaluate it
def log_reg_model (df):
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn import linear_model
    from sklearn import metrics
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    predictors = select_features(df)
    print "predictors: ", predictors
    X = df[predictors]
    Y = df['CreditStatus']

    #Split the data into training and test sets
    trainX,testX,trainY,testY = train_test_split(X, Y,
                                        test_size=0.6)
    
    clf = linear_model.LogisticRegression()
    clf.fit(trainX, trainY)
    ## expliciting the model
    md = pd.DataFrame(zip(X.columns, np.transpose(clf.coef_)))
    print "Weight coefficietns\n", md
    predicted = clf.predict(testX)
    clf.coef_
    print 'Y-axis intercept ', clf.intercept_
    print 'mean hits', np.mean(predicted == testY) ##mean hits
    print 'accuracy score', metrics.accuracy_score(testY, predicted) #scores

    #Score the training and test sets
    print "Training score: ", clf.score(trainX, trainY)
    print "Test score: ", clf.score(testX, testY)
    #KFold cross validation
    cross_val_scores = cross_val_score(linear_model.LogisticRegression(),
    X, Y, scoring='accuracy', cv=8)
    print "\n Scores with KFold cross validation: "
    print cross_val_scores
    print cross_val_scores.mean()

    return [clf, testX, testY]

# confusion matrix for a given threshold t
def classify_for_threshold (clf, X_test, Y_test, t):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    probs = clf.predict_proba(X_test)
    prob = probs[:,1]
    prob_df = pd.DataFrame(prob)
    prob_df['predict'] = np.where(prob_df[0]>=t,1,0)
    prob_df['actual'] = Y_test
    #prob_df.head()
    ## confusion matrix
    confusion_matrix=pd.crosstab(prob_df['actual'],prob_df['predict'])
    return confusion_matrix

##Generate and plot the ROC 
def gen_roc_curve(clf, X_test, Y_test):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import metrics
    
    probs = clf.predict_proba(X_test)
    prob = probs[:,1]
    prob = np.array(prob)
    Y_test = np.array(Y_test)+1
    fpr, sensitivity, _ = metrics.roc_curve(Y_test, prob, pos_label=2)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #Plot the FPR against TPR
    ax1.scatter(fpr, fpr, c='b', marker='s' )   
    ax1.scatter(fpr, sensitivity, c='r', marker='o')
    plt.show()
    #Calcuate the area under the curve
    auc = metrics.auc(fpr,sensitivity)
    print 'auc= ', auc
    return auc

def clusters(df_norm):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    ## call KMeans algo
    model = KMeans(n_clusters=10)
    model.fit(df_norm)
    ## J score
    print 'J-score = ', model.inertia_
    ## include the labels into the data
    print model.labels_
    labels = model.labels_
    md = pd.Series(labels)
    df_norm['clust'] = md
    print "\n head of norm \n",df_norm.head(5)
    ## cluster centers
    centroids = model.cluster_centers_
    centroids
    ## histogram of the clusters
    plt.hist(df_norm['clust'])
    plt.title('Histogram of Clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Frequency')
    plt.show()
    ## means of the clusters
    print "\n mean of clusters \n",df_norm.groupby('clust').mean()

    from sklearn.decomposition import PCA
    pca_data = PCA(n_components=2).fit(df_norm)
    pca_2d = pca_data.transform(df_norm)
    plt.scatter(pca_2d[:,0], pca_2d[:,1], c=labels)
    plt.title('Credit clusters')
    plt.show()

def main():
    data = clean_credit("")
    plot_column(data, "CreditStatus")
    credit_barplot(data)
    credit_boxplot(data)
    data = create_dummies(data)
    print len(data.columns.values.tolist())
    
    clf, testX, testY = log_reg_model(data)    
    print "Threshold 0.5 \n", classify_for_threshold(clf, testX, testY,0.5)
    print "Threshold 0.75 \n", classify_for_threshold(clf, testX, testY,0.75)
    print "Threshold 0.4 \n", classify_for_threshold(clf, testX, testY,0.4)
    gen_roc_curve(clf, testX, testY)
    clusters(data)

if __name__ == '__main__':
    main()