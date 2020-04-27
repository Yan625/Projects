#Alipay_Huabei
#Task objective: Estimate whether users will default in the next period



#1. Data importing
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import datetime
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
import matplotlib.pyplot as plt

data=pd.read_csv('alipay_huabei.csv')
data=pd.DataFrame.sample(data,frac=0.01,random_state=123)
data.head()#see the first five rows of dataset

#Explaination of the variables:
#limit_lab: Overdraft amount
#sex: male1,female2
#education: graduate1, undergraduate2, high school student3, other4
#marriage: married1, single2, other3
#pay_n: customer repayment situation in different period of certain year
#bil_amt1,pay_amt1: Bill amounts for different periods
#default.payment.next.month: if default in next period default1, not default0



#2. Data exploration
data.describe(include = "all")#summary of all the variables
data.dtypes#to see if there is category variables needed to make transformation into numeric variables
data.isnull().sum()# to see if there is missing value needed to be dealed with



#3. Data preprocessing
data=data.drop(['ID'], axis=1)# drop id column
x = data.ix[:,0:23]#using variables excep the last column as x
y = data.iloc[:,[23]]#using 
print(x.head())
print(y.head())



#4. Dimensions reduction--PCA
scaled_data=preprocessing.scale(x)#need to make weights not affected by variables size
pca=PCA()
pca.fit(scaled_data)
pca_data=pca.transform(scaled_data)

per_var=np.round(pca.explained_variance_ratio_*100,decimals=1)
labels=['PC'+ str(x) for x in range(1,len(per_var)+1)]

plt.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels) # check the bar graph to see how many variables can mostly explain the  target virable
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Prinnceipal Component')
plt.title("scree Plot")
plt.show()

x.corr()# to see correlation to check the multicollinearity
        #(through bar graph and correlation we decide to use pca to decrease dimensions to fixed the multicollinearity)



#5.Pipeline building | Model building
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=123)# split the data into training and testing dataset

estimator_models = [
    Pipeline(steps=[('sc',StandardScaler()),
                    ('pca',PCA()),
                    ('lr',LogisticRegression())
                    ]),
    Pipeline(steps=[('sc',StandardScaler()),
                    ('pca',PCA()),
                    ('dt',DecisionTreeClassifier()) 
                   ]),
    Pipeline(steps=[('sc',StandardScaler()),
                    ('pca',PCA()),
                    ('rf',RandomForestClassifier())
                   ]),
    Pipeline(steps=[('sc',StandardScaler()),
                    ('pca',PCA()),
                    ('svc',SVC())
                    ]),
    Pipeline(steps=[('sc',StandardScaler()),
                    ('pca',PCA()),
                    ('knn',KNeighborsClassifier())
                    ])
    ]

methods=['Lr','dt','rf', 'svm', 'knn'] 

for i,model in enumerate(estimator_models):
    model.fit(x_train, y_train)
    
    print(methods[i])
    print(model.fit(x_train, y_train))
    predict1 = model.predict(x_train)
    print(metrics.classification_report(y_train, predict1))
    cm1 = confusion_matrix(y_train, predict1)
    print(cm1)
    
    predict2 = model.predict(x_test)
    print(metrics.classification_report(y_test, predict2))
    cm2 = confusion_matrix(y_test, predict2)
    print(cm2)

#**1. Comparing the accuracy of training dataset and testing dataset we can know that five models are 
# all overfitting--training accuracy is far higer that the testing accuracy，
# we need to tun the parameter to fix the overfitting problem and meanwhile increase the accuracy.**

#**2. For example:decision tree, the accuracy of training dataset is 1 and testing only 0.67. 
# from the result we know that model's parameter:max_depth=none,max_features=none,min_samples_leaf=1, 
# to fixed the overfitting problem we need to limit the length of the tree (the longer the more easily overfit),
# the maximum number of features(the more the features, the more easily overfit), 
# and the minimum number of samples of leafs(the less the samples, the more easily overfit).**




#6. Parameter tuning¶
parameters_svm = [{'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                    'svc__C': [0.01, 0.1, 1, 10],
                    'svc__gamma': [0.001, 0.1], 
                    'svc__kernel': ['rbf']
                    },
                    {'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                    'svc__C': [0.01, 0.1, 1, 10],
                    'svc__kernel': ['linear']
                    }]

parameters_rf= {'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                'rf__n_estimators': [10, 100, 500],
                'rf__criterion': ['gini', 'entropy'], 
                'rf__max_depth': [None, 5, 8, 15, 25, 30], 
                'rf__max_features': ['sqrt', 'log2', None]
                } 

parameters_dt= {'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                 'dt__criterion': ['gini', 'entropy'], 
                 'dt__max_depth': [None, 5, 15, 25], 
                 'dt__max_features': ['sqrt', 'log2', None],
                 'dt__min_samples_leaf': [1,2,3,4,5],
                 'dt__min_samples_split':[2,3,4,5]
                 } 

parameters_lr= {'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                 "lr__penalty": ['l2'],
                 "lr__C": np.logspace(0, 4, 10),
                 "lr__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 }
            

parameters_knn= {'pca__n_components': [0.40, 0.64, 0.85, 0.95], 
                 "knn__n_neighbors": [5, 10, 15],
                 "knn__weights": ['uniform','distance'],
                 "knn__algorithm": ['auto','kd_tree','ball_tree','brute'],
                 "knn__metric":['minkowski','euclidean','manhattan','chebyshev'] 
                 }
    
parameters=[parameters_lr,parameters_dt,parameters_rf,parameters_svm,parameters_knn]
best_accuracy=0.0
best_classifier=0
pipe_dict = {0: 'Logistic Regression', 1: 'Decision Tree', 2: 'RandomForest', 3: 'SVM', 4: 'KNN'}

for i in range(len(model)):
    gridsearch = GridSearchCV(estimator_models[i], param_grid=parameters[i], cv=5) # Fit grid search
    best_model = gridsearch.fit(x_train,y_train)
    y_predict_svm = best_model.predict(x_test) 
    ts_svm = datetime.datetime.now()
    tp_svm = datetime.datetime.now() - ts_svm
    print(methods[i])
    print('training time:%s\n accuracy: %.5f\n best parameter combination: %s' % (tp_svm, gridsearch.best_score_, gridsearch.best_params_))
    print('output:\n\t%s' % classification_report(y_test, y_predict_svm, target_names=['defaut 0', 'not defaut 1']))
    if gridsearch.best_score_>best_accuracy:
        best_accuracy=gridsearch.best_score_
        best_pipeline=model
        best_classifier=i
print('Classifier with best accuracy:{}'.format(pipe_dict[best_classifier]))

#**1. After we tun the parameter, all the models increase their accuracy.**

#**2. Since the problem is to estimate whether the users will default, 
# Our accuracy in determining whether they will default is important， 
# so we choose "accuracy" as the standard for us to select models.**

#**3. svm has the most highest accuracy of 0.83 and it's trainig time is also short. 
# so we choose svm as the best model to forecast the default situation.**