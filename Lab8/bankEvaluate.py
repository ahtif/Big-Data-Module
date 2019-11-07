data = "F:/CO3093/Datasets/bank-1.csv"

def load_bank(data):
    import pandas as pd
    return pd.read_csv(data)


def create_dummies(df):
    import pandas as pd
    ## process the y to binaery 0 or 1
    df['y']=(df['y']=='yes').astype(int)
    ## create the dummies 
    cat_vars=['job','marital','education','default','housing','loan',
              'contact','month','day_of_week','poutcome']
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

def select_features(df):
    from sklearn import datasets
    import numpy as np
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    preds = df.columns.values.tolist()
    predictors = [i for i in preds if i not in ['y']]
    X0 = df[predictors]
    Y0 = df['y']
    selector = RFE(model,12,step=1) 
    selector = selector.fit(X0, Y0) 
    print(selector.support_)
    print(selector.ranking_)
    rank = selector.ranking_
    temp = np.array(predictors)
    # defining a mask
    msk = rank==1		
    s_features = temp[msk].tolist()
    return s_features

def log_reg_model (df):
    from sklearn.cross_validation import train_test_split
    from sklearn.cross_validation import cross_val_score
    from sklearn import linear_model
    from sklearn import metrics
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
    predictors = select_features(df)
    X = df[predictors]
    Y = df['y']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                       test_size=0.3, random_state=0)
    clf = linear_model.LogisticRegression()
    clf.fit(X_train, Y_train)
    ## expliciting the model
    md = pd.DataFrame(zip(X.columns, np.transpose(clf.coef_)))

    predicted = clf.predict(X_test)
    print 'mean hits', np.mean(predicted == Y_test) ##mean hits
    print 'accuracy score', metrics.accuracy_score(Y_test, predicted) #scores
   ### cross validation scores
    scores = cross_val_score(linear_model.LogisticRegression(), X, Y,
             scoring='accuracy', cv=8)
    print 'crosss valiadation mean scores:', scores.mean()

    return [clf, X_test, Y_test]

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

## ROC curve
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
    ax1.scatter(fpr, fpr, c='b', marker='s' )
    ax1.scatter(fpr, sensitivity, c='r', marker='o')
    plt.show()
    auc = metrics.auc(fpr,sensitivity)
    print 'auc= ', auc
    return auc
    
 
