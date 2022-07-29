import numpy as np
import os
import cv2
import sys
from tqdm import tqdm
import time
import matplotlib.image as mpimg

#locate where the output will be saved in
os.chdir(sys.argv[2])



data_dir= sys.argv[1] #"/home/khaled/Downloads/Burn_mod/" #dir for the images
CATEGORIES=["Partial_thickness_burn","Normal_skin","Full_thickness_burn"] # creat a list for labiling  creat a list for labiling
IMG_SIZE=100

training_data = []
rotate_img=[]
part_ev=[]
part=[]
normal_ev=[]
normal=[]
full_ev=[]
full=[]
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

def to_one_hot(labels, dimension=3):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
         results[i, label] = 1.
    return results

def create_training_data():
    imeges=[]

    for num in range (0,360,15):
        for category in CATEGORIES:  # do Normal skin, Partial thickness burn or full thickness burn
            if category == "Partial_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  # get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image in partial thickniss burn
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        extraction= cv2.imread(os.path.join(path,img))
                        part.append([new_array, class_num, extraction])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))

            elif category == "Normal_skin":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image in normal skin
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        extraction= cv2.imread(os.path.join(path,img))
                        normal.append([new_array, class_num,extraction])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))

            elif category == "Full_thickness_burn":
                path = os.path.join(data_dir,category)  # create path to the burns catigories
                class_num = CATEGORIES.index(category)  #get the classification  (0 , 1 or 2). 0= Partial_thickness_burn 1=Normal_skin 2= Full_thickness_burn
                for img in tqdm(os.listdir(path)):  # iterate over each image per Full thickniss burn
                    try:
                        img_array = mpimg.imread(os.path.join(path,img))  # convert to array
                        rotate_img=rotate_bound(img_array,num)
                        new_array = cv2.resize(rotate_img, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                        extraction= cv2.imread(os.path.join(path,img))
                        full.append([new_array, class_num, extraction])
                    except Exception as e:  # in the interest in keeping the output clean...
                        pass
                        #except OSError as e:
                            #print("OSErrroBad img most likely", e, os.path.join(path,img))
                                #except Exception as e:
                            #print("general exception", e, os.path.join(path,img))



create_training_data() # run the created function to extact the data




# shuffle the data

import random
random.shuffle(part)
random.shuffle(normal)
random.shuffle(full)


# extract the evaluation data from the training data
part_ev=part[0:100]

part=part[100:]

normal_ev=normal[0:100]
normal=normal[100:]

full_ev=full[0:100]
full=full[100:]


training_data=part+normal+full
random.shuffle(training_data)

test_data=part_ev+normal_ev+full_ev
random.shuffle(test_data )




# extract the x (training data) and y (labels) to introduce them to the model
#x for feture y for label

x=[]
y=[]


for features, label, ex in training_data:#the list has tow part first is the image the second is the label
    x.append(features)
    y.append(label)
#because we cant feed a list into the CNN we have to convert it into array
y = to_one_hot(y)
x=np.array(x).reshape(-1,IMG_SIZE,IMG_SIZE,3) #-1 is the numper of features, 1 is the gray scale  #this is were the tensor created

x = x/255.0

part_x=[]
part_y=[]
part_extract=[]
for features, label, ex in part_ev:#the list has tow part first is the image the second is the label
    part_x.append(features)
    part_y.append(label)
    part_extract.append(ex)
#because we cant feed a list into the CNN we have to convert it into array
part_hot_y = to_one_hot(part_y)
part_x=np.array(part_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)



part_x=part_x/255.0

nor_x=[]
nor_y=[]
nor_extract=[]
for features, label, ex in normal_ev:#the list has tow part first is the image the second is the label
    nor_x.append(features)
    nor_y.append(label)
    nor_extract.append(ex)
#because we cant feed a list into the CNN we have to convert it into array
nor_hot_y = to_one_hot(nor_y)
nor_x=np.array(nor_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
nor_x=nor_x/255.0

full_x=[]
full_y=[]
full_extract=[]
for features, label, ex in full_ev:#the list has tow part first is the image the second is the label
    full_x.append(features)
    full_y.append(label)
    full_extract.append(ex)
#because we cant feed a list into the CNN we have to convert it into array
full_hot_y = to_one_hot(full_y)
full_x=np.array(full_x).reshape(-1,IMG_SIZE,IMG_SIZE,3)
full_x=full_x/255.0


x_test=[]
y_test=[]
for features, label, ex in test_data:#the list has tow part first is the image the second is the label
    x_test.append(features)
    y_test.append(label)
#because we cant feed a list into the CNN we have to convert it into array
y_test = to_one_hot(y_test)
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

pickle_out = open(os.path.join(folder,"Partial_burned_skin_x.pickle"),"wb")
pickle.dump(part_x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Partial_hot_burned_skin_y.pickle"),"wb")
pickle.dump(part_hot_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Partial_burned_skin_y.pickle"),"wb")
pickle.dump(part_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Extract_partial.pickle"),"wb")
pickle.dump(part_extract, pickle_out)
pickle_out.close()


pickle_out = open(os.path.join(folder,"Normal_skin_x.pickle"),"wb")
pickle.dump(nor_x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Hot_normal_skin_y.pickle"),"wb")
pickle.dump(nor_hot_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Normal_skin_y.pickle"),"wb")
pickle.dump(nor_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Extract_normal.pickle.pickle"),"wb")
pickle.dump(nor_extract, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Full_burned_skin_x.pickle"),"wb")
pickle.dump(full_x, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Full_hot_burned_skin_y.pickle"),"wb")
pickle.dump(full_hot_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Full_burned_skin_y.pickle"),"wb")
pickle.dump(full_y, pickle_out)
pickle_out.close()

pickle_out = open(os.path.join(folder,"Extract_full.pickle.pickle"),"wb")
pickle.dump(full_extract, pickle_out)
pickle_out.close()



