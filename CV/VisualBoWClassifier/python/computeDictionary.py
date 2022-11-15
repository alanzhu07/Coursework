import pickle
from getDictionary import get_dictionary


meta = pickle.load(open('../data/traintest.pkl', 'rb'))
train_imagenames = meta['train_imagenames']

# -----fill in your implementation here --------
# alpha, K = 50, 100
alpha, K = 100, 300
dictionary_random = get_dictionary(train_imagenames, alpha, K, 'Random')
print(dictionary_random.shape)
with open('dictionaryRandom.pkl', 'wb') as fh:
    pickle.dump(dictionary_random, fh)

dictionary_harris = get_dictionary(train_imagenames, alpha, K, 'Harris')
print(dictionary_harris.shape)
with open('dictionaryHarris.pkl', 'wb') as fh:
    pickle.dump(dictionary_harris, fh)

# ----------------------------------------------



