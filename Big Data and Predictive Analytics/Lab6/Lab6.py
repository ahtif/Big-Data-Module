## Create pair-wise scatter plots
def auto_pairs(plot_cols, df):
	import matplotlib.pyplot as plt
	from pandas.tools.plotting import scatter_matrix
	fig = plt.figure(1, figsize=(12, 12))
	fig.clf()
	ax = fig.gca()
	scatter_matrix(df[plot_cols], alpha=0.3,
			diagonal='kde', ax = ax)
	plt.show()
	return('Done')

def data_split(df, test_size=0.3):
	from sklearn.cross_validation import train_test_split
	# train, test = train_test_split(df, test_size)
	train = df.sample(frac=1-test_size,random_state=200)
	test = df.drop(train.index)
	return train, test

def main():
	import auto_price_clean as auto

	df = auto.clean_auto("")

	# print "Before normalization \n"
	# print df.describe()

	df = auto.normalize(df)

	# print "\n After normalization \n"
	# print df.describe()
	plot_cols = ["curb-weight", "engine-size"]
	# auto_pairs(plot_cols, df)

	# Assume you have named your cleaned dataset 'auto_price'
	# drop the column 'price'
	df.drop(['price'], axis = 1, inplace = True)
	

	df_train, df_test = data_split(df)
	print ("Training size: {}, Testing size: {}".format(len(df_train),len(df_test)))
	print ("Samples: {}, Features: {}".format(*df_train.shape))
	model(df)

def model(auto_price):
	# Run this code interactively for a better understanding
	# Assume your clean data is in df
	import matplotlib.pyplot as plt
	import numpy as np
	from sklearn.linear_model import LinearRegression
	from sklearn.cross_validation import train_test_split
	from sklearn.feature_selection import RFE
	from sklearn.svm import SVR
	# Assume your clean data is in auto_price
	df = auto_price
	# all columns of the data
	feature_cols = df.columns.values.tolist()
	# remove the outcome 'lnprice'
	feature_cols.remove('lnprice')
	X0 = df[feature_cols]
	Y0 = df['lnprice']
	estimator = SVR(kernel="linear")
	# We desire a linear model with 5 variables
	# If we want 2 features, we change 5 to 2; etc.
	selector = RFE(estimator,5,step=1)
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
	trainX,testX,trainY,testY = train_test_split(X,Y, test_size = 0.2)

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
	plt.plot(xrange(int(df['lnprice'].max())),
	xrange(int(df['lnprice'].max())),color='darkorange')
	plt.xlabel = "Actual Prices"
	plt.ylabel = "Predicted Prices"
	plt.title("Plot of predicted vs actual prices")
	plt.show()


if __name__ == '__main__':
	main()