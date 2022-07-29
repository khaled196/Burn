import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
from scipy import interp
from itertools import cycle
from sklearn.metrics import roc_curve, auc


# IN-DIR
os.chdir(sys.argv[1])

data= os.path.join(sys.argv[1],'data')



model = tf.keras.models.load_model('Burn_images.model')


# test_data
pickle_in = open(os.path.join(data,"x_test.pickle"),"rb")
x_test = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"y_test.pickle"),"rb")
y_test = pickle.load(pickle_in)



pickle_in = open(os.path.join(data,"Partial_burned_skin_x.pickle"),"rb")
part_x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Partial_hot_burned_skin_y.pickle"),"rb")
part_hot_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Partial_burned_skin_y.pickle"),"rb")
part_y = pickle.load(pickle_in)

#Full thickness burn

pickle_in = open(os.path.join(data,"Full_burned_skin_x.pickle"),"rb")
full_x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Full_hot_burned_skin_y.pickle"),"rb")
full_hot_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Full_burned_skin_y.pickle"),"rb")
full_y = pickle.load(pickle_in)


pickle_in = open(os.path.join(data,"Normal_skin_x.pickle"),"rb")
nor_x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Hot_normal_skin_y.pickle"),"rb")
nor_hot_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Normal_skin_y.pickle"),"rb")
nor_y = pickle.load(pickle_in)


# extract 

pickle_in = open(os.path.join(data,"Extract_partial.pickle"),"rb")
part_extract = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Extract_normal.pickle"),"rb")
nor_extract = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Extract_full.pickle"),"rb")
full_extract = pickle.load(pickle_in)




CATEGORIES=["Part_burned_skin","Normal_skin","Full_burned_skin"] # creat a list for labiling  creat a list for labiling


def extract(image)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([0, 53, 104])
  upper = np.array([179, 240, 255])
  mask = cv2.inRange(hsv, lower, upper)
  result = cv2.bitwise_and(burned_skin_x[i], burned_skin_x[i], mask=mask)
  return result
  


  # OUT-DIR
os.chdir(sys.argv[2])




# partial evaluation
print("partial_thicknes_evaluation")
result = model.predict(part_x)

model.evaluate(part_x,part_hot_y)
for i in range(len(part_y)):



    if part_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        #font_color = 'red' if part_y[i] != np.argmax(result[i]) else 'black'
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        #plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(part_x[i],cmap='gray')
        fig.savefig('part_miss_predicted'+str(i)+ '.png')
        plt.close()
        extract(part_extract[i])
        cv2.imwrite('extracted_mispredicted_partial_burn_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)

    elif part_y[i] == np.argmax(result[i]):
        fig= plt.figure()
        plt.title("the correct catigory is "+CATEGORIES[int(part_y[i])])
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(part_y[i])], fontsize=18, color="black")
        plt.imshow(part_x[i],cmap='gray')
        plt.savefig('part_correct_predicted'+str(i)+ '.png')
        plt.close()
        extract(part_extract[i])
        cv2.imwrite('extracted_correct_predicted_partial_burn_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)

# normal skin evaluation
print("normal_skin_evaluation")
result = model.predict(nor_x)

model.evaluate(nor_x,nor_hot_y)

for i in range(len(nor_y)):
    if nor_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        #font_color = 'red' if nor_y[i] != np.argmax(result[i]) else 'black'
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        #plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_miss_predicted'+str(i)+ '.png')
        plt.close()
        extract(nor_extract[i])
        cv2.imwrite('extracted_mispredicted_normal_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)
    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(nor_y[i])])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(nor_y[i])], fontsize=18, color="black")
        plt.imshow(nor_x[i],cmap='gray')
        plt.savefig('nor_correct_predicted'+str(i)+ '.png')
        plt.close()
        extract(nor_extract[i])
        cv2.imwrite('extracted_correctly_predected_normal_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)

# full thickness evaluate
print("full_thicknes_evaluation")
result = model.predict(full_x)


model.evaluate(full_x,full_hot_y)

for i in range(len(full_y)):
    if full_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])
        #font_color = 'red' if full_y[i] != np.argmax(result[i]) else 'black'
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        #plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_miss_predicted'+str(i)+ '.png')
        plt.close()
        extract(full_extract[i])
        cv2.imwrite('extracted_mispredicted_full_burned_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)


    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(full_y[i])])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(full_y[i])], fontsize=18, color="black")
        plt.imshow(full_x[i],cmap='gray')
        plt.savefig('full_correct_predicted'+str(i)+ '.png')
        plt.close()
        extract(full_extract[i])
        cv2.imwrite('extracted_correctly_predicted_full_burned_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)










#importing confusion matrix
y_pred = model.predict(x_test)
y_pred_1=np.argmax(y_pred, axis=1)

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test, y_pred_1)
print('Confusion Matrix\n')
print(confusion)


array = confusion
df_cm = pd.DataFrame(array, index = [i for i in ["Partial_thickness_burn","Normal_skin","Full_thickness_burn"]],
                  columns = [i for i in ["Partial_thickness_burn","Normal_skin","Full_thickness_burn"]])
fig= plt.figure(figsize = (10,7))
fig.patch.set_facecolor('xkcd:white')
sn.heatmap(df_cm, annot=True)
plt.savefig('Confusion_matrix.png')





#Results ROC CURVE




pickle_in = open(os.path.join(data,"hot_y_test.pickle"),"rb")
y_test = pickle.load(pickle_in)


y_score = model.predict(x_test)
# Plot linewidth.
lw = 2
n_classes=3
classes=["Partial_thickness_burn","Normal_skin","Full_thickness_burn" ]
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig= plt.figure(1)
fig.patch.set_facecolor('xkcd:white')
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of  {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.png')



# Zoom in view of the upper left corner.
fig= plt.figure(2)
fig.patch.set_facecolor('xkcd:white')
plt.xlim(0, 0.2)
plt.ylim(0.8, 1)
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of {0} (area = {1:0.2f})'
             ''.format(classes[i], roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic curve (upper left corner)')
plt.legend(loc="lower right")
plt.savefig('Zoomed.png')






