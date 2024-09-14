  #LOGISTIC REGRESSION
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB



#STEP 2 : create instances of imported ones
logr=LogisticRegression(random_state=0)
gb=GradientBoostingClassifier(n_estimators=10)
dt=DecisionTreeClassifier(random_state=0)
rf= RandomForestClassifier(random_state=0)
sv=svm.SVC()
nn=MLPClassifier(solver='lbfgs', alpha = 1e-5,hidden_layer_sizes=(5,2),random_state=0)
mnb=MultinomialNB()
gn=GaussianNB()




df=pd.read_csv("D:/Mrunmayee Naik/IRIS.csv")

#STEP 3: IDENTIFYING X AND Y
 # X= the features and Y= the label on basis here supervised learning is performed

#Now remove y from x by drop funct put axis =1 this removes whole column
X=df.drop("species",axis =1 ) 
#select the y axis
Y=df["species"]

print(X)

print(Y)

#STEP 4: SPLITING THE DATA FOR TRAINING AND TESTING
  #ORDER TO BE FOLOWED IS CONSTANT

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,random_state=0,test_size=0.3)
#RANDOM STATE  bcz it shuffles so the acc score should be same , traintest funct shuffles the df

#STEP 5 : TO TRAIN BY APPLYING THE ALGO
train= logr.fit(X_train,Y_train)
train=gb.fit(X_train,Y_train)
train=dt.fit(X_train,Y_train)
train=rf.fit(X_train,Y_train)
train=sv.fit(X_train,Y_train)
train=nn.fit(X_train,Y_train)
train=mnb.fit(X_train,Y_train)
train=gn.fit(X_train,Y_train)


#STEP 6 :PREDICTION
y_pred= logr.predict(X_test)
y1_pred=gb.predict(X_test)
y2_pred=dt.predict(X_test)
y3_pred=rf.predict(X_test)
y4_pred=sv.predict(X_test)
y5_pred=nn.predict(X_test)
y6_pred=mnb.predict(X_test)
y7_pred=gn.predict(X_test)

#STEP 7 : CHECK ACCURACY SCORE Y_TEST = ACTUAL VALUE y_pred= predicted value
print("Logistic Regression",accuracy_score(Y_test,y_pred))
print("Gradient Booster",accuracy_score(Y_test,y1_pred))
print("Decision tree",accuracy_score(Y_test,y2_pred))
print("Random Forest",accuracy_score(Y_test,y3_pred))
print("Support Vector",accuracy_score(Y_test,y4_pred))
print("MLP Classifier",accuracy_score(Y_test,y5_pred))
print("MultinomiAL nb",accuracy_score(Y_test,y6_pred))
print("Gaussian Nb",accuracy_score(Y_test,y7_pred))
#accuracy score cannot be 100%

#accuracy
'''Logistic Regression 0.9777777777777777
Gradient Booster 0.9777777777777777
Decision tree 0.9777777777777777
Random Forest 0.9777777777777777
Support Vector 0.9777777777777777
MLP Classifier 0.24444444444444444
MultinomiAL nb 0.6
Gaussian Nb 1.0 '''
