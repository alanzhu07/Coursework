import numpy as np
import cv2
#Import necessary functions
import skimage.transform
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH

#Write script for Q4.1
cap_src = cv2.VideoCapture('../data/ar_source.mov')
cap_book = cv2.VideoCapture('../data/book.mov')
cv_cover = cv2.imread('../data/cv_cover.jpg')

# frames = loadVid('../data/book.mov')
# src_frames = loadVid('../data/ar_source.mov')

counter = 0
frames = []
while cap_book.isOpened():
    ret, frame = cap_book.read()
    if ret:
        frames.append(frame)
        counter += 1
    else:
        cap_book.release()

print(frames[-1].shape)
print(counter)

height, width, _ = cv_cover.shape

cap_src = cv2.VideoCapture('../data/ar_source.mov')
counter = 0
src_frames = []
while cap_src.isOpened():
    ret, frame = cap_src.read()
    if ret:
        frame = frame[40:320, 208:431]
        src_frames.append(frame)
        counter += 1
    else:
        cap_src.release()    
print(src_frames[-1].shape)
print(counter)

print(cv_cover.shape)

# Default resolutions of the frame.
frame_width = 640
frame_height = 480

# Define the codec and create VideoWriter object.  The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('ar.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

for frame_number, frame in enumerate(frames):
    img = src_frames[frame_number % len(src_frames)]
    img = skimage.transform.resize(img, (height, width))
    matches, locs1, locs2 = matchPics(cv_cover, frame, sigma=0.1, ratio=0.75)
    x_cv_cover = locs1[matches[:,0]][:,[1,0]]
    x_frame = locs2[matches[:,1]][:,[1,0]]
    H_cover_to_frame, inliers = computeH_ransac(x_frame, x_cv_cover, num_iters=300, threshold=5)
    composite = compositeH(H_cover_to_frame, frame, img)
    print(f"frame {frame_number}, number of matches {matches.shape[0]}, number of inliers {inliers.sum()}")

    out.write(np.uint8(composite))

out.release()
