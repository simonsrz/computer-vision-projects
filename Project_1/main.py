import os
import cv2
import numpy as np

sum_dice = 0  # sum of all dice coefficient - to calculate mean
sum_iou = 0  # sum of all dice coefficient - to calculate mean
count = 0  # segmented images counter
arr = []  # array that stores images with their dice coefficient - to choose the best and worst ones


# iteration over images from /segmented directory
for filename in os.listdir('segmented'):
    # reading segmented image to variable
    seg = cv2.imread(os.path.join('segmented', filename))
    # if there is such an image
    if seg is not None:
        # changing color space of segmented image to grayscale
        gray = cv2.cvtColor(seg, cv2.COLOR_RGB2GRAY)

        # binary thresholding
        T, thresh = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY)

        # removing noice inside leaf's contours
        mask_gt = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))

        # reading color image corresponding to the current segmented one tand assign it to variable
        img = cv2.imread('color/' + filename[:len(filename) - 17] + '.JPG')

        # changing color space of image to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # setting brown and green hsv color ranges
        lower_brown = np.array([0, 95, 86])
        upper_brown = np.array([31, 255, 200])

        lower_green = np.array([25, 37, 33])
        upper_green = np.array([120, 255, 255])

        # creating two masks using the above color ranges
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # connecting two masks into one
        mask = cv2.bitwise_or(green_mask, brown_mask)

        # removing noise and empty spots with morphology and blur
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))
        mask = cv2.medianBlur(mask, 7)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), dtype=np.uint8))

        # calculating dice coefficient and IoU
        intersect = np.sum(cv2.bitwise_and(mask, mask_gt))
        sum1 = np.sum(mask)
        sum2 = np.sum(mask_gt)

        dice = (2 * intersect) / (sum1 + sum2)
        iou = intersect / np.sum(cv2.bitwise_or(mask, mask_gt))

        # appending leaf to array
        arr.append([dice, iou, 'color/' + filename[:len(filename) - 17] + '.JPG'])

        # things necessary to obtain mean results
        sum_iou = sum_iou + iou
        sum_dice = sum_dice + dice
        count = count + 1

        # writing obtained mask to the file
        cv2.imwrite(os.path.join('masks', filename[:len(filename) - 17] + '.JPG'), mask)


# sorting array of masks and print 5 worst and 5 best according to dice coefficient
arr = sorted(arr, key=lambda l: l[0])
print(arr[:5])
print(arr[len(arr)-5:len(arr)])

# printing mean results
print('Dice coefficient:')
print(sum_dice/count)
print('Intersection over union:')
print(sum_iou/count)

