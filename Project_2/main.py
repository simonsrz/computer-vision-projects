import cv2
import skimage.feature as skifeat
import numpy as np
import os
import mahotas
from sklearn.ensemble import RandomForestClassifier

# https://gogulilango.com/software/image-classification-python
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# https://pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
def fd_lbp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = skifeat.local_binary_pattern(gray, 24, 8, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, 24 + 3),
                             range=(0, 24 + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    # return the histogram of Local Binary Patterns
    return hist


train_path = "train"
valid_path = "valid"
train_filenames = os.listdir(train_path)
test_filenames = os.listdir(valid_path)

global_features = []
train_labels = []

for training_name in train_filenames:
    # join the training directory with the texture name
    dir = os.path.join(train_path, training_name)
    # get the current training texture label
    current_label = training_name[4:]
    # loop over the images in each texture subfolder
    for img in os.listdir(dir):
        # image file name
        file = dir + "/" + str(img)
        # reading image
        image = cv2.imread(file)

        # Here we pick the feature that we want to test and comment
        # the remaining ones:

        feature = fd_haralick(image)
        # feature = fd_histogram(image)
        # feature = np.hstack([fd_haralick(image), fd_histogram(image)])
        # feature = fd_lbp(image)

        # appending current texture label and its feature to corresponding arrays
        train_labels.append(current_label)
        global_features.append(feature)


# defining classifier model
clf = RandomForestClassifier(n_estimators=100, random_state=9)
# training classifier model
clf.fit(global_features, train_labels)


test_labels = []
total_hits = 0

for testing_name in test_filenames:
    # join the testing directory with the texture name
    dir = os.path.join(valid_path, testing_name)
    # get the current training texture label
    current_label = testing_name[4:]
    hits = 0
    count = 0
    for image in os.listdir(dir):
        # appending correct label of the texture to the array
        test_labels.append(current_label)
        # image file name
        file = dir + "/" + str(image)
        # reading file
        img = cv2.imread(file)

        # Here we pick the same feature as in the previous loop
        # and comment the remaining ones:

        feature = fd_haralick(img)
        # feature = fd_histogram(img)
        # feature = np.hstack([fd_haralick(img), fd_histogram(img)])
        # feature = fd_lbp(img)

        # using pretrained classifier to predict the label of the texture
        prediction = clf.predict(feature.reshape(1, -1))[0]

        count += 1
        # calculating accuracy percentages:
        if prediction == current_label:
            total_hits += 1
            hits += 1
    print(str(current_label) + ' correct percentage: ', 100*hits/count)

print('Percentage correct: ', 100*total_hits/len(test_labels))



