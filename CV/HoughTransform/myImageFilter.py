import numpy as np

def myImageFilter(img0, h, mode='edge'):
    # h is assumed to be in odd dimensions
    filter_height, filter_length = h.shape
    img_height, img_length = img0.shape
    pad_verti, pad_horiz = (filter_height - 1) // 2, (filter_length - 1) // 2
    img_padded = np.pad(img0, ((pad_verti, pad_verti), (pad_horiz, pad_horiz)), mode=mode)
    h = np.flip(h)

    img = np.zeros(img0.shape)

    for height in range(filter_height):
        for length in range(filter_length):
            img += img_padded[height:height+img_height, length:length+img_length] * h[height, length]

    return img