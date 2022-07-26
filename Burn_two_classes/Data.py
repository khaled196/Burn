import numpy as np
import os
import cv2
import sys
from tqdm import tqdm
import time
import matplotlib.image as mpimg

#locate where the output will be saved in
os.chdir(sys.argv[2])


data_dir= sys.argv[1] # images folders location
CATEGORIES=["burned_skin","Normal_skin"] # creat a list for labiling  creat a list for labiling
IMG_SIZE=75

burned_skin = []
normal_skin=[]
rotate_img=[]

def to_one_hot(labels, dimension=2):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
         results[i, label] = 1.
    return results


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))




def create_training_data():
    for num in range (0,360,15):
        for category in CATEGORIES:  # do Normal skin or Burned skin

            if category == "Normal_skin":
                path = os.path.join(data_dir,category)  # create path to the skin catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1). 0= Burned skin 1=Normal_skin
                for img in tqdm(os.listdir(path)):  # iterate over each image in normal skin
                    try:
                        img_array = cv2.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        normal_skin.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))

            elif category == "burned_skin":
                path = os.path.join(data_dir,category)  # create path to the skin catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1). 0= Burned skin 1=Normal_skin
                for img in tqdm(os.listdir(path)):  # iterate over each image in Burned skin
                    try:
                        img_array = cv2.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        burned_skin.append([new_array, class_num])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))


create_training_data()


#shuffle the data to randomise it

import random
random.shuffle(burned_skin)
random.shuffle(normal_skin)


# extract the evaluation data from the training data
burned_skin_ev=burned_skin[0:100]

burned_skin=burned_skin[100:]

normal_skin_ev=normal_skin[0:100]
normal_skin=normal_skin[100:]

test_data= normal_skin_ev+burned_skin_ev
random.shuffle(test_data)

#create training data
training_data=burned_skin+normal_skin

random.shuffle(training_data)

#x for feture y for label
x=[]
y=[]

for sample in training_data:#the list has tow part first is the image the second is the label
    x.append(sample[0])
    y.append(sample[1])
#because we cant feed a list into the CNN we have to convert it into array

x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3) #-1 refere that we have a large numper of the sampels we got, 3 refeare that we have colored images  #this is were the tensor created
y=to_one_hot(y)

x = x/255.0


# Burned skin 
burned_skin_x=[]
burned_skin_y=[]
for features, label in burned_skin_ev:#the list has tow part first is the image the second is the label
    burned_skin_x.append(features)
    burned_skin_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
hot_burned_skin_y= to_one_hot(burned_skin_y)

burned_skin_x=np.array(burned_skin_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)


#burned_skin_x=burned_skin_x/255.0



# Normal skin 
normal_skin_x=[]
normal_skin_y=[]
for features, label in normal_skin_ev:#the list has tow part first is the image the second is the label
    normal_skin_x.append(features)
    normal_skin_y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
hot_normal_skin_y= to_one_hot(normal_skin_y)
normal_skin_x=np.array(normal_skin_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)

#normal_skin_x=normal_skin_x/255.0

# Test data
x_test=[]
y_test=[]
for features, label in test_data:#the list has tow part first is the image the second is the label
    x_test.append(features)
    y_test.append(label)
#because we cant feed a list into the CNN we have to convert it into array
x_test=np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,3)
x_test=x_test/255.0


import pickle

folder= os.path.join(sys.argv[2],'data')
if not os.path.exists(folder):
    os.mkdir(folder)
pickle_out = open(os.path.join(folder,"x.pickle"),"wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"y.pickle"),"wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"x_test.pickle"),"wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"y_test.pickle"),"wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"burned_skin_x.pickle"),"wb")
pickle.dump(burned_skin_x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"hot_burned_skin_y.pickle"),"wb")
pickle.dump(hot_burned_skin_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"burned_skin_y.pickle"),"wb")
pickle.dump(burned_skin_y, pickle_out)
pickle_out.close()


pickle_out = open(os.path.join(folder,"normal_skin_x.pickle"),"wb")
pickle.dump(normal_skin_x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"hot_normal_skin_y.pickle"),"wb")
pickle.dump(hot_normal_skin_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"normal_skin_y.pickle"),"wb")
pickle.dump(normal_skin_y, pickle_out)
pickle_out.close()


