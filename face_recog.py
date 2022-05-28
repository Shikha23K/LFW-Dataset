from  sklearn.datasets import fetch_lfw_people
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

faces= fetch_lfw_people(min_faces_per_person=50)

print("All face data size : ", faces.data.shape)
# 1560 samples and 2914 are features
print("Each face data size : ", faces.images[0].shape)
# 62 X 47 sized matrix
print("Names of target people: ",faces.target_names)

print(faces.target_names.size)

print(np.unique(faces.target))


#model built
pcaModel=PCA(n_components=150,whiten=True)
svmModel=SVC(kernel='rbf',class_weight='balanced')
mdl=make_pipeline(pcaModel,svmModel)

#splitting data into training and validation
from sklearn.model_selection import train_test_split
X_train,X_Test,yTrain,yTest=train_test_split(faces.data,faces.target,test_size=.20)

#training or model to data
 
from sklearn.model_selection import GridSearchCV
param_grid={'svc__C':[1,5,15,30],'svc__gamma':[0.00001,0.00005,0.0001,0.005]}
grid=GridSearchCV(mdl,param_grid)
grid.fit(X_train,yTrain)

print(grid.best_params_)
mdl=grid.best_estimator_

#each target converted into numerical code as 0 to 11


#for indx, axindx in enumerate(ax.flat):
 #   axindx.imshow(faces.images[indx],cmap='bone')
   # axindx.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[indx]])

#plt.show()

y_pred=mdl.predict(X_Test)
fig, ax=plt.subplots(6,8)
for indx, axindx in enumerate(ax.flat):
    axindx.imshow(X_Test[indx].reshape(62,47),cmap='bone')
    axindx.set(xticks=[],yticks=[])
    axindx.set_ylabel(faces.target_names[y_pred[indx]].split()[-1],color='green'if y_pred[indx]==yTest[indx] else 'red')
    fig.suptitle('Wrong in red!' ,size=14)
    
plt.show()

#accuracy
from sklearn.metrics import classification_report
print(classification_report(yTest,y_pred,target_names=faces.target_names))

#confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
mat=confusion_matrix(yTest,y_pred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,xticklabels=faces.target_names)
plt.xlabel("True Label")
plt.ylabel("predicted Label")
plt.show()