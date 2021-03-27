#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


stars_df = pd.read_csv('Star_type.csv')


# In[3]:


df1 = pd.read_csv('Star_type.csv')


# In[4]:


Type = []
for i in range(len(stars_df['Star type'])):
    if stars_df['Star type'][i] == 0:
        Type.append('Brown dwarf')
    if stars_df['Star type'][i] == 5:
        Type.append('Hypergiant')
    if stars_df['Star type'][i] == 3:
        Type.append('Main sequence')
    if stars_df['Star type'][i] == 1:
        Type.append('Red dwarf')
    if stars_df['Star type'][i] == 4:
        Type.append('Supergiant')
    if stars_df['Star type'][i] == 2:
        Type.append('White dwarf')


# In[5]:


stars_df['Type'] = Type


# In[6]:


stars_df


# In[7]:


brown_dwarf = len(stars_df.loc[stars_df['Star type'] == 0])
red_dwarf = len(stars_df.loc[stars_df['Star type'] == 1])
white_dwarf = len(stars_df.loc[stars_df['Star type'] == 2])
main_sequence = len(stars_df.loc[stars_df['Star type'] == 3])
supergiant = len(stars_df.loc[stars_df['Star type'] == 4])
hypergiant = len(stars_df.loc[stars_df['Star type'] == 5])

print("Brown dwarf = {} ".format(brown_dwarf))
print("Red dwarf  = {} ".format(red_dwarf))
print("White Dwarf = {} ".format(white_dwarf))
print("Main Sequence = {} ".format(main_sequence))
print("Supergiant= {} ".format(supergiant))
print("Hypergiant = {} ".format(hypergiant)) 
print("Total stars in the dataset = {} ".format(len(stars_df)))


# In[8]:


print(stars_df['Star color'].unique())


# In[9]:


x = stars_df['Spectral Class'].unique()
y = stars_df['Spectral Class'].value_counts()
plt.figure(figsize=(20,8))
sns.barplot(x,y)


# In[10]:


stars_df['Star color'].value_counts()


# In[11]:


stars_df['Spectral Class'].value_counts()


# In[12]:


figure= plt.figure(figsize=(10,10))
sns.distplot(stars_df['Temperature (K)'],bins=10)


# In[13]:


def func(pct, allvalues): 
    absolute = int(pct / 100.*np.sum(allvalues)) 
    return "({:d} stars )".format(absolute) 
data = [brown_dwarf,red_dwarf,white_dwarf,main_sequence,supergiant,hypergiant]

plt.figure(figsize=(10,10))
x = stars_df['Type'].value_counts()
y = stars_df['Type'].unique()
c = ['indigo','blue','green','yellow','orange','red']
plt.pie(x,labels=y,colors=c,startangle=90,shadow=True,explode=(0.1,0,0.1,0,.1,0),autopct= lambda pct: func(pct, data))
plt.show()


# In[14]:


sns.jointplot(y='Absolute magnitude(Mv)',x='Temperature (K)',data=stars_df, kind='hex')


# In[15]:


fig = plt.figure(figsize=(10,8))
x1 = np.array(df1.drop(['Star color','Spectral Class'],1))
y1 = np.array(df1['Star type'], dtype ='float')
y1.shape = (len(y1),1)
c1 =0

for i in range(0,len(x1)):
    if x1[i][4] == 0:
        a = plt.scatter(x1[i][0],x1[i][3], s = 30 , c = 'green', marker = '+')
    elif x1[i][4]== 1:
        b = plt.scatter(x1[i][0],x1[i][3],s = 50 , c = 'red',marker = '^')
    elif x1[i][4]== 2:
        c = plt.scatter(x1[i][0],x1[i][3],s = 75 , c = 'gray',marker = 'x')
    elif x1[i][4]== 3:
        d = plt.scatter(x1[i][0],x1[i][3],s = 90 , c = 'brown',marker = 'o')     
    elif x1[i][4]== 4:
        e = plt.scatter(x1[i][0],x1[i][3],s = 100 , c = 'orange',marker = '*') 
    elif x1[i][4]== 5:
        f = plt.scatter(x1[i][0],x1[i][3],s = 120 , c = 'blue',marker = ',')
        
        
    c1+=1


print("Total Counted Stars = {}".format(c1)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Total Stars ")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.plot(5778,4.83,marker='o',markersize=15,label = 'SUN',markerfacecolor = 'yellow',markeredgewidth=3,
         markeredgecolor='black')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.text(8200,5.83,'SUN')
plt.show()


# In[16]:


x = np.array(df1.drop(['Star type', 'Star color','Spectral Class'],1))   
y = np.array(df1['Star type'], dtype ='float')                           
y.shape = (len(y),1)


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.3)


# In[18]:


from sklearn import preprocessing, model_selection
x_f_train = preprocessing.scale(x_train)
x_f_test = preprocessing.scale(x_test)
y_f_train = y_train
y_f_test = y_test


# In[19]:


fig = plt.figure(figsize=(10,8))

c2=0

for i in range(0,len(y_train)):
    if y_train[i] == 0:
        a = plt.scatter(x_train[i][0],x_train[i][3], s = 30 , c = 'green', marker = '+')
    elif y_train[i]== 1:
        b = plt.scatter(x_train[i][0],x_train[i][3],s = 50 , c = 'red',marker = '^')
    elif y_train[i]== 2:
        c = plt.scatter(x_train[i][0],x_train[i][3],s = 75 , c = 'gray',marker = 'x')
    elif y_train[i]== 3:
        d = plt.scatter(x_train[i][0],x_train[i][3],s = 90 , c = 'brown',marker = 'o')      
    elif y_train[i]== 4:
        e = plt.scatter(x_train[i][0],x_train[i][3],s = 100 , c = 'orange',marker = '*') 
    elif y_train[i]== 5:
        f = plt.scatter(x_train[i][0],x_train[i][3],s = 120 , c = 'blue',marker = ',')    
    c2+=1


print("Total Trained Stars = {}".format(c2)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Trained Stars ")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# In[20]:


fig = plt.figure(figsize=(10,8))
c3=0

for i in range(0,len(y_test)):
    if y_test[i] == 0:
        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '+')
    elif y_test[i]== 1:
        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '^')
    elif y_test[i]== 2:
        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = 'x')
    elif y_test[i]== 3:
        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = 'o')   
    elif y_test[i]== 4:
        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = '*')
    elif y_test[i]== 5:
        f = plt.scatter(x_test[i][0],x_test[i][3],s = 120 , c = 'blue',marker = ',')     
    c3+=1


print("Total Tested Stars = {}".format(c3)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Tested Stars ")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# # KNN CLASSIFICATION

# In[21]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[22]:


for i in list(stars_df.columns):
    if stars_df[i].dtype=='object':
        stars_df[i]=le.fit_transform(stars_df[i])


# In[23]:


x = np.array(df1.drop(['Star type', 'Star color','Spectral Class'],1))   
y = np.array(df1['Star type'], dtype ='float').reshape(len(y),)                          


# In[24]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=11,test_size=0.2)


# In[25]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
accu_list=[]
for i in range(1,11):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_s=knn.predict(x_test)
    scores=accuracy_score(y_test,pred_s)
    accu_list.append(scores)


# In[26]:


accu_list


# In[27]:


from sklearn.metrics import classification_report,confusion_matrix
print('CONFUSION MATRIX \n\n',confusion_matrix(y_test,pred_s))
print('\n\n__________________________________________________________________________\n\n')
print('CLASSIFICATION REPORT \n\n',classification_report(y_test,pred_s))


# In[28]:


plt.plot(range(1,11),accu_list,marker='o',markerfacecolor = 'red')
plt.xlabel('K Values')
plt.ylabel('Accuracy Scores')
plt.show()


# In[29]:


max(accu_list)


# In[56]:


knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
pred_s=knn.predict(x_test)
scores=accuracy_score(y_test,pred_s)


# In[57]:


fig = plt.figure(figsize=(10,8))
c4=0

for i in range(0,len(pred_s)):
    if pred_s[i] == 0:
        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '+')
    elif pred_s[i]== 1:
        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '^')
    elif pred_s[i]== 2:
        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = 'x')
    elif pred_s[i]== 3:
        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = 'o')   
    elif pred_s[i]== 4:
        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = '*')
    elif pred_s[i]== 5:
        f = plt.scatter(x_test[i][0],x_test[i][3],s = 120 , c = 'blue',marker = ',')     
    c4+=1


print("Total Tested Stars = {}".format(c4)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Predicted Stars by KNN")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# # LOGISTIC REGRESSION

# In[31]:


from sklearn.linear_model import LogisticRegression


# In[35]:


accu_list_1 = []
for i in range(10000,21000,1000):
    lr=LogisticRegression(max_iter=i,multi_class='auto',solver = 'lbfgs')
    lr.fit(x_train,y_train)
    pred_1=lr.predict(x_test)
    score_1=accuracy_score(y_test,pred_1)
    accu_list_1.append(score_1)


# In[33]:


accu_list_1


# In[36]:


from sklearn.metrics import classification_report,confusion_matrix
print('CONFUSION MATRIX \n\n',confusion_matrix(y_test,pred_1))
print('\n\n__________________________________________________________________________\n\n')
print('CLASSIFICATION REPORT \n\n',classification_report(y_test,pred_1))


# In[37]:


fig = plt.figure(figsize=(10,8))
c5=0

for i in range(0,len(pred_1)):
    if pred_1[i] == 0:
        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '+')
    elif pred_1[i]== 1:
        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '^')
    elif pred_1[i]== 2:
        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = 'x')
    elif pred_1[i]== 3:
        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = 'o')   
    elif pred_1[i]== 4:
        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = '*')
    elif pred_1[i]== 5:
        f = plt.scatter(x_test[i][0],x_test[i][3],s = 120 , c = 'blue',marker = ',')     
    c5+=1


print("Total Tested Stars = {}".format(c5)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Predicted Stars by Logistic regression")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# # RANDOM FOREST CLASSIFIER

# In[43]:


from sklearn.ensemble import RandomForestClassifier


# In[44]:


accu_list_2 = []
for i in range(100,200,10):
    rfc=RandomForestClassifier(n_estimators=i)
    rfc.fit(x_train,y_train)
    pred_2=rfc.predict(x_test)
    score_2=accuracy_score(y_test,pred_2)
    accu_list_2.append(score_2)


# In[45]:


accu_list_2


# In[46]:


from sklearn.metrics import classification_report,confusion_matrix
print('CONFUSION MATRIX \n\n',confusion_matrix(y_test,pred_2))
print('\n\n__________________________________________________________________________\n\n')
print('CLASSIFICATION REPORT \n\n',classification_report(y_test,pred_2))


# In[47]:


fig = plt.figure(figsize=(10,8))
c6=0

for i in range(0,len(pred_2)):
    if pred_2[i] == 0:
        a = plt.scatter(x_test[i][0],x_test[i][3], s = 30 , c = 'green', marker = '+')
    elif pred_2[i]== 1:
        b = plt.scatter(x_test[i][0],x_test[i][3],s = 50 , c = 'red',marker = '^')
    elif pred_2[i]== 2:
        c = plt.scatter(x_test[i][0],x_test[i][3],s = 75 , c = 'gray',marker = 'x')
    elif pred_2[i]== 3:
        d = plt.scatter(x_test[i][0],x_test[i][3],s = 90 , c = 'brown',marker = 'o')   
    elif pred_2[i]== 4:
        e = plt.scatter(x_test[i][0],x_test[i][3],s = 100 , c = 'orange',marker = '*')
    elif pred_2[i]== 5:
        f = plt.scatter(x_test[i][0],x_test[i][3],s = 120 , c = 'blue',marker = ',')     
    c6+=1


print("Total Tested Stars = {}".format(c6)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Predicted Stars by Random Forest Classifier")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# # SUPPORT VECTOR MACHINES (SVM)

# In[48]:


from sklearn.svm import SVC
svm=SVC(gamma='auto',)


# In[49]:


svm.fit(x_train,y_train)
pred_3=svm.predict(x_test)
score_3=accuracy_score(y_test,pred_3)


# In[50]:


score_3


# In[156]:


# I think there is no need to plot a graph for this model since the accuracy score is very very low


# In[157]:


# from all the classifiers used logistic regression and random forest gives the highest accuracy scores


# In[158]:


new_stars_df=pd.DataFrame({'actual':y_test,'predictions':pred_2})


# In[159]:


new_stars_df


# # DECISION TREE

# In[170]:


stars_df = pd.read_csv('Star_type.csv')


# In[171]:


Spectral = pd.get_dummies(stars_df['Spectral Class'],drop_first=True)
stars_df.drop('Spectral Class',axis=1,inplace=True)


# In[172]:


stars_df = pd.concat([stars_df,Spectral],axis=1)


# In[173]:


stars_df = stars_df.drop('Star color',axis=1)


# In[174]:


stars_df.head()


# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(np.array(stars_df.drop('Star type',axis=1)),
                                                    np.array(stars_df['Star type'],dtype ='float').reshape(len(y),1),
                                                    test_size=0.30,random_state=101)


# In[59]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
dtree= DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions= dtree.predict(X_test)
score=metrics.accuracy_score(y_test,predictions)
print('Score = ',score)


# In[60]:


from sklearn.metrics import classification_report,confusion_matrix
print('CONFUSION MATRIX \n\n',confusion_matrix(y_test,predictions))
print('\n\n__________________________________________________________________________\n\n')
print('CLASSIFICATION REPORT \n\n',classification_report(y_test,predictions))


# In[178]:


fig = plt.figure(figsize=(10,8))
c7=0

for i in range(0,len(predictions)):
    if predictions[i] == 0:
        a = plt.scatter(X_test[i][0],X_test[i][3], s = 30 , c = 'green', marker = '+')
    elif predictions[i]== 1:
        b = plt.scatter(X_test[i][0],X_test[i][3],s = 50 , c = 'red',marker = '^')
    elif predictions[i]== 2:
        c = plt.scatter(X_test[i][0],X_test[i][3],s = 75 , c = 'gray',marker = 'x')
    elif predictions[i]== 3:
        d = plt.scatter(X_test[i][0],X_test[i][3],s = 90 , c = 'brown',marker = 'o')   
    elif predictions[i]== 4:
        e = plt.scatter(X_test[i][0],X_test[i][3],s = 100 , c = 'orange',marker = '*')
    elif predictions[i]== 5:
        f = plt.scatter(X_test[i][0],X_test[i][3],s = 120 , c = 'blue',marker = ',')     
    c7+=1


print("Total Tested Stars = {}".format(c7)) 
plt.xlabel("Temperature(K)")
plt.ylabel("Absolute Magnitude(Mv)")
plt.title("H-R Diagram of Predicted Stars by Decision Tree")
plt.legend((a,b,c,d,e,f),('Brown Dwarf','Red Dwarf','White Dwarf','Main Sequence','Supergiant','Hypergiant'))
m = np.linspace(0,40000,100)
n = (0*m) -7.5
plt.plot(m,n,'--m')
f = np.linspace(0,6000,100)
g = (0*f)+15.4
plt.plot(f,g,'--c')
plt.gca().invert_xaxis()
plt.gca().invert_yaxis()
plt.show()


# In[179]:


from IPython.display import Image
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz
import pydot


# In[180]:


df_feat = stars_df.drop('Star type',axis=1)
features = list(df_feat.columns)


# In[181]:


dot_data=StringIO()
export_graphviz(dtree,out_file=dot_data,feature_names=features,filled=True,rounded=True)


# In[182]:


graph = pydot.graph_from_dot_data(dot_data.getvalue())  
Image(graph[0].create_png())


# In[ ]:




