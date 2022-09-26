#!/usr/bin/python
import time

def task4(dictionary, raw):
	"""
	TODO:
		implement your optimized edit distance function for task 4 here
		dictionary : path of dictionary.txt file 
		raw: path of raw.txt file
		return : a list of min_distance of each word in the raw.txt 
		compared with words in the dictonary 
	example return result : [0,1,0,2]
	"""
	with open(dictionary, 'r') as dict_file:
		dict_words = [word for word in dict_file.read().split()]

	## construct trie = [l, b, trie]
	## where l is the level, b indicate if there is a word end in the character
	dict_trie = [0, False, {}]
	height = 0
	for word in dict_words:
		level = dict_trie
		for i in range(len(word)):
			if word[:i+1] not in level[2]:
				level[2][word[:i+1]] = [level[0]+1, False, {}]
				height = max(height, level[0]+1)
			level = level[2][word[:i+1]]
			if i is len(word) - 1:
				level[1] = True
	
	# print(dict_trie)
	# print("Trie height %d" % height)
	
	with open(raw, 'r') as raw_file:
		raw_words = [word for word in raw_file.read().split()]
	
	def dfs(trie, word, key, table, all_distances):
		## update table column corresponding to the level
		for i in range(1, len(table)):
			j = trie[0]
			str1 = word
			str2 = key
			if j is 0:
				table[i][j] = i
			else:
				del_cost = table[i][j-1] + 1
				ins_cost = table[i-1][j] + 1
				sub_cost = table[i-1][j-1] if str1[i-1] is str2[j-1] else table[i-1][j-1]+1
				trans_cost = table[i-2][j-2] + 1 if i > 1 and j > 1 and str1[i-2] is str2[j-1] and str1[i-1] is str2[j-2] else None
				table[i][j] = min(del_cost, ins_cost, sub_cost, trans_cost) if trans_cost else min(del_cost, ins_cost, sub_cost)
		
		## reach a root/end of word
		if trie[1] is True:
			## finish calculate distance for one word in trie, add to list to compare
			all_distances.append(table[len(word)][trie[0]])
		
		## go to the next level
		if trie[2] is not {}:
			for key in trie[2]:
				dfs(trie[2][key], word, key, table, all_distances)

	distances = []

	## iterate through raw words
	for word in raw_words:
		distance_for_word = []

		## construct a table with len(word)+1 rows
		table = [[row for col in range(height+1)] for row in range(len(word)+1)]
		# print('Original table: {}'.format(table))

		## dfs - update table
		dfs(dict_trie, word, '', table, distance_for_word)
		distances.append(min(distance_for_word))
		

	return distances

if __name__ == '__main__':
	c = task4('test.txt', 'rawtest.txt')
	t1 = time.time()
	c = task4('dictionary.txt', 'raw.txt')
	t2 = time.time()
	print(c)
	print(t2 - t1)