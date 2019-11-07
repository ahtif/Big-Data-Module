def pairedCount(word1,word2,fpath):
	import re, string
	fhand = open(fpath, 'r') #open file in reading mode
	count1 = 0
	count2 = 0
	# Generate regex to match only the words
	regex1 = r'\b'+word1+r'\b'
	regex2 = r'\b'+word2+r'\b'
	for line in fhand:
		# Remove punctuation from the line
		line = line.translate(None, string.punctuation)
		# Count the occurences of each word in the line
		count1 += len(re.findall(regex1, line, re.IGNORECASE | re.U))
		count2 += len(re.findall(regex2, line, re.IGNORECASE | re.U))
	# Close the file once we are done with it
	fhand.close()
	# (Used for testing) print each word and it's number of occurrences
	print (word1, count1) , (word2, count2)
	# Return the minimum amount of occurences for the pair
	return count1 if (count1 <= count2) else count2

def crossCorrelation1(wordList, fpath):
	import itertools as it
	# Generate every possible combination of pairs from the list
	combinations = it.combinations(wordList,2)
	pairOccurrences = dict()
	# Loop through every possible pair of words
	for x, y in combinations:
		# Set the value of each pair in the dictionary to the minimum number of occurences in the file
		pairOccurrences[(x, y)] = pairedCount(x, y, fpath)
	# Return a dictionary containing each pair of words and its number of occurences
	return pairOccurrences

def crossCorrelation2(wordList, fileList):
	import itertools as it
	# Generate every possible combination of pairs from the list
	combinations = it.combinations(wordList,2)
	pairOccurrences = dict()
	for x, y in combinations:
		# Create a counter to add up the number of occurences of the pair in all fils
		count = 0
		# Loop through each file and add the occurences of the pair to the sum
		for file in fileList:
			count += pairedCount(x, y, file)
		# Set the value of each pair in the dictionary to the minimum number of occurences in the files
		pairOccurrences[(x, y)] = count
	# Return a dictionary containing each pair of words and its number of occurences
	return pairOccurrences

def test():
	assert pairedCount("algorithm","matrix", "A1Test/paper1.txt") == 14
	assert pairedCount("differentiation", "algorithm", "A1Test/paper1.txt") == 15
	assert pairedCount("source transformation", "computer", "comp_terms_list.txt") == 1
	assert pairedCount("diff", "node", "words_list.txt") == 0
	
	d1 = crossCorrelation1(["matrix","algorithm","differentiation"],"A1Test/paper1.txt")
	assert d1[("matrix","algorithm")] == 14
	assert d1[("matrix","differentiation")] == 14
	assert d1[("algorithm","differentiation")] == 15
	
	d2 = crossCorrelation1(["matrix","algorithm","differentiation"],"A1Test/paper2.txt")
	assert d2[("matrix","algorithm")] == 3
	assert d2[("matrix","differentiation")] == 3
	assert d2[("algorithm","differentiation")] == 4	

	d3 = crossCorrelation2(["matrix","algorithm","differentiation"],["A1Test/paper1.txt","A1Test/paper2.txt"])
	assert d3[("matrix","algorithm")] == 17
	assert d3[("matrix","differentiation")] == 17
	assert d3[("algorithm","differentiation")] == 19

	print "All tests passed!"

if __name__ == '__main__':
	test()
