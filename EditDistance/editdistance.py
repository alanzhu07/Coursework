#!/usr/bin/python

class EditDistance():
	
	def __init__(self):
		"""
		Do not change this
		"""
	
	def calculateLevenshteinDistance(self, str1, str2):
		"""
		TODO:
			take two strings and calculate their Levenshtein Distance for task 1 
			return an integer which is the distance

        Hint:
	 	This approach uses dynamic programming. Key idea is if we compute the distance 
	 	between the prefix of the first string and the prefix of the second string, 
	 	find the distance between the two full strings using the distance between the last
	 	computed prefix values. Store the distances in a matrix to incrementally
	 	solve the problem, one character at a time.

	 	You are free to use other approaches to the problem.
		Pseudocode:
		Create a 2D distance table with:
			rows = length of str1 +1
			col = length of str2 +1

		for i goes from 1 to number of rows in table:
			for j goes from 1 to number of col in table:
				1. Compute delete_cost

				2. Compute insertion_cost

				3. Compute substitute_cost

				Store minimum of the 3 costs at the table[i][j] location.

		return table[row][col]

		"""

		table = [[[] for col in range(len(str2)+1)] for row in range(len(str1)+1)]
		for i in range(len(table)):
			for j in range(len(table[0])):
				if i is 0 or j is 0:
					table[i][j] = max(i, j)
				else:
					del_cost = table[i][j-1] + 1
					ins_cost = table[i-1][j] + 1
					sub_cost = table[i-1][j-1] if str1[i-1] is str2[j-1] else table[i-1][j-1]+1
					table[i][j] = min(del_cost, ins_cost, sub_cost)
		return table[len(table)-1][len(table[0])-1]

		
	def calculateOSADistance(self, str1, str2):
		"""
		TODO:
			take two strings and calculate their OSA Distance for task 2 
			return an integer which is the distance

		Hint: Use the Pseudocode provided in calculateLevenshteinDistance and 
		add an additional check to compute the transpose distance. Update 
		the table[i][j] location's value accordingly.

		"""
		
		table = [[[] for col in range(len(str2)+1)] for row in range(len(str1)+1)]
		for i in range(len(table)):
			for j in range(len(table[0])):
				if i is 0 or j is 0:
					table[i][j] = max(i, j)
				else:
					del_cost = table[i][j-1] + 1
					ins_cost = table[i-1][j] + 1
					sub_cost = table[i-1][j-1] if str1[i-1] is str2[j-1] else table[i-1][j-1]+1
					trans_cost = table[i-2][j-2] + 1 if i > 1 and j > 1 and str1[i-2] is str2[j-1] and str1[i-1] is str2[j-2] else None
					table[i][j] = min(del_cost, ins_cost, sub_cost, trans_cost) if trans_cost else min(del_cost, ins_cost, sub_cost)
		return table[len(table)-1][len(table[0])-1]
		
	def calculateDLDistance(self, str1, str2):
		"""
		TODO:
			take two strings and calculate their DL Distance for task 3 
			return an integer which is the distance

		Pseudocode:
		Create a 2D distance table with:
			rows = length of str1 +1
			columns = length of str2 +1

		Create an alphabet_map:
			A dictionary of unique characters of str1 & str2:
				key: unique character
				value: Number of unique characters seen before the 'key'
			Eg. str1 = 'abc', str2 = 'man'
			alphabet_map = {'a':0, 'b': 1, 'c': 2, 'm':3, 'n': 4}

		Create distance1 = list of 0s of length as number entries in alphabet_map

		for i goes from 1 to number of rows in table:
			Reset distance2 to 0
			for j goes from 1 to number of cols in table:
				k = distance1[alphabet_map[str2[j - 1]]]
				l = distance2
				1. Compute delete_cost

				2. Compute insertion_cost

				3. Compute substitute_cost
					Additionally, if substitute is possible
						Set distance2 to j

				4. Compute transposition_cost
					Sum of the following: 
					a. table[k - 1][l - 1]
					b. (i - k - 1)
					c. (j - l - 1)
					d. 1

				Store minimum of the 4 costs at the table[i][j] location.

			Set distance1[alphabet_map[str1[i - 1]]] to i

		return table[row][col]

		"""
		table = [[[] for col in range(len(str2)+1)] for row in range(len(str1)+1)]
		alphabet_map = {}
		unique = 0
		for c in range(len(str1+str2)):
			if (str1+str2)[c] not in alphabet_map:
				alphabet_map[(str1+str2)[c]] = unique
				unique += 1
		distance1 = [0 for s in range(len(alphabet_map))]
		
		for i in range(len(table)):
			distance2 = 0
			for j in range(len(table[0])):
				if i is 0 or j is 0:
					table[i][j] = max(i, j)
				else:
					k = distance1[alphabet_map[str2[j-1]]]
					l = distance2
					del_cost = table[i][j-1] + 1
					ins_cost = table[i-1][j] + 1
					sub_cost = table[i-1][j-1]
					if str1[i-1] is not str2[j-1]:
						sub_cost += 1
						distance2 = j
					trans_cost = (i-k-1) + (j-l-1) + 1 if k is 0 or l is 0 else table[k-1][l-1] + (i-k-1) + (j-l-1) + 1
					table[i][j] = min(del_cost, ins_cost, sub_cost, trans_cost)
			if i > 0:
				distance1[alphabet_map[str1[i-1]]] = i
	
		return table[len(table)-1][len(table[0])-1]


if __name__ == '__main__':
	c = EditDistance()
	word1 = input("Enter word 1:")
	word2 = input("Enter word 2:")
	print("OSA:{}, DL:{}".format(c.calculateOSADistance(word1, word2), c.calculateDLDistance(word1, word2)))