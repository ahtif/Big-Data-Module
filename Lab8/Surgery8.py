def main():
	import pandas as pd
	import matplotlib.pyplot as plt
	
	df = pd.read_csv('wine.csv')
	# plot the column quality
	plt.hist(df['quality'])
	plt.title('Wine Quality')
	plt.show()
	print plt.clf()
	# means of chemichal composition
	print df.groupby('quality').mean()
	## scater plot of alcohol vs quality
	plt.plot(df['alcohol'], df['quality'],'ro')
	plt.title('Qual vs Alco')
	plt.xlabel('Alcohol')
	plt.ylabel('Quality')
	plt.show()
	## normalisation
	df_norm = (df - df.min()) / (df.max() - df.min())
	print "\n\n", df.head(2)
	print "\n\n", df_norm.head(2)

	from sklearn.cluster import KMeans
	## call KMeans algo
	model = KMeans(n_clusters=6)
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
	plt.title('Wine clusters')
	plt.show()

if __name__ == '__main__':
	main()