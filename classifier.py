# ================================ imports  ==================================================#
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import math
import sys
import glob
import os
from os import path
from skimage import feature
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import petl as etl
from tabulate import tabulate
# ====================================================================================================#

# ================================ open image ==================================================#
def open_img(_path):
    img = cv2.imread(_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)), plt.title('result')
    # plt.show()
    return gray
# ====================================================================================================#

# ================================ get image feature  ==================================================#
def get_image_feature(img):
    numPoints = 8
    radius = 1

    # extract the histogram of Local Binary Patterns
    lbp = feature.local_binary_pattern(img, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))
    # optionally normalize the histogram
    eps = 1e-7
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    return hist
# ====================================================================================================#

# ================================ build data set  ==================================================#
def build_data_set(imeges_data_path):
    male = imeges_data_path + "/male"
    male_images_list = glob.glob(male + "/*.jpg")
    female = imeges_data_path + "/female"
    female_images_list = glob.glob(female + "/*.jpg")

    data2 = []
    for im in male_images_list:
        img = open_img(im)
        img_feature = get_image_feature(img)
        data2.append({"label": 1, "feature": img_feature, "gender": "male"})

    for im in female_images_list:
        img = open_img(im)
        img_feature = get_image_feature(img)
        data2.append({"label": 0, "feature": img_feature, "gender": "female"})

    df = pd.DataFrame(data2)

    return df
# ====================================================================================================#

# ================================ train model svc  ==================================================#
def train_model_svc(train_dataset, test_dataset):
    train_data = list(train_dataset['feature'])
    train_labels = list(train_dataset['label'])
    test_data = list(test_dataset['feature'])
    test_labels = list(test_dataset['label'])

    # Create a linear SVM classifier
    #clf = svm.SVC(kernel="linear")

    # Create a rfb SVM classifier
    clf = svm.SVC(kernel='rbf', C=1, gamma=10)

    # Train classifier
    clf.fit(train_data, train_labels)

    # Make predictions on unseen test data
    clf_predictions = clf.predict(test_data)

    print("Accuracy: {}%".format(clf.score(test_data, test_labels) * 100))
    # printing report and different statistics
    #print(classification_report(test_labels, clf.predict(test_data)))

    return clf_predictions
# ====================================================================================================#

# ================================ Create result file  ==================================================#
def create_result_file(y_true, y_pred):
    f = open("result.txt", "w")
    f.write("Best kernel: RFP \n\n")
    f.write("Best Parameters:\n\n  Radios = 1 points_num = 8 \t C = 1 gamma = 10 \n\n")
    f.write("  Radios = 3 points_num = 24 \t C = 0.1 gamma = 0.1 \n\n")
    f.write("Best Accuracy: 68.57%\n\n\n")


    f.write("  Confusion Matrix\n")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    records = [('male', tp, fn), ('female', fp, tn)]
    df = pd.DataFrame.from_records(records, columns=(' ', 'male', 'female'))
    table = etl.fromdataframe(df)
    f.write(tabulate(table))

    f.close()
# ====================================================================================================#

# ================================ main  ==================================================#
if __name__ == '__main__':
    train_dataset = build_data_set(sys.argv[1])    #"gender_split/train"
    test_dataset = build_data_set(sys.argv[3])     #"gender_split/test"
    #valid_dataset = build_data_set(sys.argv[2])   #"gender_split/valid"

    predictions = train_model_svc(train_dataset, test_dataset)
    create_result_file(test_dataset['label'], predictions)
# linear (1,8)                           valid - 66.66%  test- 67.14%
# linear (3,24)                          valid - 65.27%  test- nan
# RFB (1,8) (c=1,g=10)                   valid - 66.66%  test- 68.57%
# RFB (3,24) (c=0.1,g=0.1)               valid - 66.66%  test- 68.57%
