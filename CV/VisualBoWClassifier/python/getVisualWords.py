import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    filter_responses = extract_filter_responses(I, filterBank).reshape(-1, 3*len(filterBank))
    dists = cdist(filter_responses, dictionary, 'euclidean')
    wordMap = dists.argmin(axis=1).reshape(I.shape[0], I.shape[1])

    # ----------------------------------------------

    return wordMap

if __name__ == '__main__':
    import cv2 as cv
    import pickle
    from createFilterBank import create_filterbank
    import skimage
    dictionary_harris = pickle.load(open('dictionaryHarris.pkl', 'rb'))
    dictionary_random = pickle.load(open('dictionaryRandom.pkl', 'rb'))
    filterBank = create_filterbank()

    img_paths = [
        '../data/bedroom/sun_aacyfyrluprisdrx.jpg',
        '../data/landscape/sun_aawnncfvjepzpmly.jpg',
        '../data/campus/sun_abslhphpiejdjmpz.jpg'
    ]

    for i, img_path in enumerate(img_paths):
        im = cv.cvtColor(cv.imread(img_path), cv.COLOR_BGR2RGB)
        wordMap = get_visual_words(im, dictionary_harris, filterBank)
        visual_words = skimage.color.label2rgb(wordMap)
        cv.imwrite(f"../output/visual_Harris{i+1}.jpg", cv.cvtColor((255*visual_words).astype(np.uint8), cv.COLOR_RGB2BGR))
        wordMap = get_visual_words(im, dictionary_random, filterBank)
        visual_words = skimage.color.label2rgb(wordMap)
        cv.imwrite(f"../output/visual_Random{i+1}.jpg", cv.cvtColor((255*visual_words).astype(np.uint8), cv.COLOR_RGB2BGR))

