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
import cv2
# IN-DIR
os.chdir(sys.argv[1])

data= os.path.join(sys.argv[1],'data')



model = tf.keras.models.load_model('Burn_images.model')


# test_data
pickle_in = open(os.path.join(data,"x_test.pickle"),"rb")
x_test = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"y_test.pickle"),"rb")
y_test = pickle.load(pickle_in)



pickle_in = open(os.path.join(data,"burned_skin_x.pickle"),"rb")
burned_skin_x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"hot_burned_skin_y.pickle"),"rb")
hot_burned_skin_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"burned_skin_y.pickle"),"rb")
burned_skin_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"normal_skin_x.pickle"),"rb")
normal_skin_x = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"hot_normal_skin_y.pickle"),"rb")
hot_normal_skin_y = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"normal_skin_y.pickle"),"rb")
normal_skin_y = pickle.load(pickle_in)

#extract

pickle_in = open(os.path.join(data,"Extract_burned.pickle"),"rb")
burned_extract = pickle.load(pickle_in)

pickle_in = open(os.path.join(data,"Extract_normal.pickle"),"rb")
normal_extract = pickle.load(pickle_in)









CATEGORIES=["burned_skin","Normal_skin"] # creat a list for labiling  creat a list for labiling


def extract(image)
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  lower = np.array([0, 50, 0])
  upper = np.array([179, 240, 255])
  mask = cv2.inRange(hsv, lower, upper)
  result = cv2.bitwise_and(burned_skin_x[i], burned_skin_x[i], mask=mask)
  return result

# OUT-DIR
os.chdir(sys.argv[2])

# partial evaluation
print("Burned Skin evaluation")
result = model.predict(burned_skin_x)

model.evaluate(burned_skin_x,hot_burned_skin_y)
for i in range(len(burned_skin_y)):
    if burned_skin_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        #font_color = 'red' if burned_skin_y[i] != np.argmax(result[i]) else 'black'
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(burned_skin_y[i])], fontsize=18, color="black")
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(burned_skin_y[i])])
        #plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(burned_skin_x[i],cmap='gray')
        fig.savefig('mispredicted_burn_skin'+str(i)+ '.png')
        plt.close()
        extract(burned_extract[i])
        cv2.imwrite('extracted_mispredicted_burn_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)


    elif burned_skin_y[i] == np.argmax(result[i]):
        fig= plt.figure()
        plt.title("the correct catigory is "+CATEGORIES[int(burned_skin_y[i])])
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(burned_skin_y[i])], fontsize=18, color="black")
        plt.imshow(burned_skin_x[i],cmap='gray')
        plt.savefig('burnd_correct_predicted'+str(i)+ '.png')
        plt.close()
        extract(burned_extract[i])
        cv2.imwrite('extracted_correctly_predicted_burn_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)


# normal skin evaluation
print("normal_skin_evaluation")


result = model.predict(normal_skin_x)

model.evaluate(normal_skin_x,hot_normal_skin_y)

for i in range(len(normal_skin_y)):
    if normal_skin_y[i] != np.argmax(result[i]):
        fig= plt.figure()
        #font_color = 'red' if normal_skin_y[i] != np.argmax(result[i]) else 'black'
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(normal_skin_y[i])])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(normal_skin_y[i])], fontsize=18, color="black")
        #plt.text(x=40, y=-10, s=CATEGORIES[int(np.argmax(result[i]))], fontsize=18, color=font_color)
        plt.imshow(normal_skin_x[i],cmap='gray')
        plt.savefig('nor_miss_predicted'+str(i)+ '.png')
        plt.close()
        extract(normal_extract[i])
        cv2.imwrite('extracted_mispredicted_normal_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)

    else:
        fig= plt.figure()
        fig.suptitle("the predected catigory is "+CATEGORIES[int(np.argmax(result[i]))])
        plt.title("the correct catigory is "+CATEGORIES[int(normal_skin_y[i])])
        #plt.text(x=-20, y=-10, s=CATEGORIES[int(normal_skin_y[i])], fontsize=18, color="black")
        plt.imshow(normal_skin_x[i],cmap='gray')
        plt.savefig('nor_correct_predicted'+str(i)+ '.png')
        plt.close()
        extract(normal_extract[i])
        cv2.imwrite('extracted_correctly_predicted_normal_skin'+str(i)+ '.png', result)
        cv2.waitKey(0)




print("Burned Skin evaluation")
model.evaluate(burned_skin_x,hot_burned_skin_y)

print("normal_skin_evaluation")
model.evaluate(normal_skin_x,hot_normal_skin_y)





#importing confusion matrix
y_pred = model.predict(x_test)
y_pred_1=np.argmax(y_pred, axis=1)


confusion = confusion_matrix(y_test, y_pred_1)
print('Confusion Matrix\n')
print(confusion)


array = confusion
df_cm = pd.DataFrame(array, index = [i for i in ["Burned","Healthy"]],
                  columns = [i for i in ["Burned","Healthy"]])
fig= plt.figure(figsize = (10,7))
fig.patch.set_facecolor('xkcd:white')
sn.heatmap(df_cm, annot=True)
plt.savefig('Confusion_matrix.png')






pickle_in = open(os.path.join(data,"hot_y_test.pickle"),"rb")
y_test = pickle.load(pickle_in)


#Results ROC CURVE
y_score = y_pred
# Plot linewidth.
lw = 2
n_classes=2
classes=["Burned skin","Normal skin"]
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




