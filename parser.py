import urllib.request
import csv
import os
import cv2

# Download images with George:
# i = 1
# with open('georges.csv', 'r') as file_george:
#     reader = csv.reader(file_george)
#     for row in reader:
#         url = row[0]
#         urllib.request.urlretrieve(url, 'data/george/' + str(i) + '.jpg')
#         print(f'Downloaded image № {i}')
#         i += 1

# Download images without George:
# i = 1
# with open('non_georges.csv', 'r') as file_non_george:
#     reader = csv.reader(file_non_george)
#     for row in reader:
#         url = row[0]
#         urllib.request.urlretrieve(url, 'data/non_george/' + str(i) + '.jpg')
#         print(f'Downloaded image № {i}')
#         i += 1

train_num = 1
test_num = 1
# George Folder:
for i in range(1, 2682):
    # Move to Train Folder
    if i % 5 != 0:
        image = cv2.imread('data/george/' + str(i) + '.jpg')
        image = cv2.resize(image, (256, 256))
        cv2.imwrite('data/train/' + str(train_num) + '.jpg', image)
        train_num += 1
    # Move to Test Folder every 5th image
    else:
        image = cv2.imread('data/george/' + str(i) + '.jpg')
        image = cv2.resize(image, (256, 256))
        cv2.imwrite('data/test/' + str(test_num) + '.jpg', image)
        test_num += 1
    print(f'George Image № {i}')
george_train_last = train_num
george_test_last = test_num
# Non-George Folder:
for i in range(1, 3367):
    # Move to Train Folder
    if i % 5 != 0:
        image = cv2.imread('data/non_george/' + str(i) + '.jpg')
        image = cv2.resize(image, (256, 256))
        cv2.imwrite('data/train/' + str(train_num) + '.jpg', image)
        train_num += 1
    # Move to Test Folder every 5th image
    else:
        image = cv2.imread('data/non_george/' + str(i) + '.jpg')
        image = cv2.resize(image, (256, 256))
        cv2.imwrite('data/test/' + str(test_num) + '.jpg', image)
        test_num += 1
    print(f'Non-George Image № {i}')

print(f'Last Train Image George: {george_train_last}')
print(f'Last Test Image George: {george_test_last}')