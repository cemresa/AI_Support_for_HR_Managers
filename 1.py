# importing libraries and methods
# sklearn will be used for the ML methods and evaluation metrics
# matplotlib and seaborn are visualisation libraries
# pandas is used for the dataframe
# numpy is used for mathematical calculations

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# %% 

# SECTION 1 - READING DATASET

# reading daatset
df = pd.read_csv("C:/Users/MONSTER/Desktop/AI_PROJE/Kod/1.csv", sep=",")
# %%
# we need information about the dataset
df.info()
# dropping the contact info columns
# these won't be needed in the training steps.

df_contact_info = df.drop(df.iloc[:, 3:24], axis=1)
df = df.drop(columns=["name", "surname", "mail"])
# %%

# the dataset consists of string values
# we need them to be double, float or int
# in order to implement ML methods.
# by printing out the unique values
# in each column, we can arrange the dataset.

for col in df:
    print("Number of unique values in column '" +
          str(col) + "' is " + str(len(df[col].unique())))

for col in df:
    print(df[col].unique())
# %%

# the strings are converted into int values.

# in school column, there are many different values
# we proposed a method as follows
# the HR will choose the universities they are accepting
# the values will be appended to a list.
# the selected universities are represented by 1
# the others are represented as 0


# normally, the desired universities will be chosen by the HR manager
# since we couldn't do that, this is the method we chose

selected_uni = ['Middle East Technical University',
                'Koc University', 'Yildiz Technical University',
                'Bogazici University', 'Istanbul University-Cerrahpasa',
                'Istanbul Technical University', 'Istanbul University',
                'Sabanci University', 'Hacettepe University']
df.loc[df['school'].isin(selected_uni), 'school'] = 1
other_uni = df['school'].unique().tolist()
other_uni.pop(0)
df.loc[df['school'].isin(other_uni), 'school'] = 0

# in the other case, the code as follows

"""
try:
    selected_uni = []
      
    while True:
        selected_uni.append(int(input()))
          
except:
    print(selected_uni)

"""

# the universities would be chosen from the dropdown menu
# and the indexes of the respective selection would be appended
# to the selected_uni list
# this method would be applied to each column.
# %%

# there are too many string values in
# programs column too
# this column is neglected in our case
# because we couldn't derive a solution

df.drop(columns=["programs"], inplace=True)
# %%

# in city_living column, there are 62 unique values.
# print(len(df.city_living.unique()))
# we proposed a method as follows
# the HR will choose the cities in which they have offices
# the values will be appended to a list.
# the selected cities are represented by 1
# the others are represented as 0

selected_city = ['Ankara', 'Istanbul', 'Eskisehir', 'Kocaeli', 'Izmir', 'Mersin',
                 'Bursa', 'Antalya']
df.loc[df['city_living'].isin(selected_city), 'city_living'] = 1
other_city = df['city_living'].unique().tolist()
other_city.pop(0)
df.loc[df['city_living'].isin(other_city), 'city_living'] = 0
# %%

# the rest of the columns do not consist that many values

# gender column consists of 2 char values
# 1 represents male, 0 represents female
# alone_group column also consists of 2 char values
# 1 represents group, 0 represent alone

df.gender = [1 if each == 'M' else 0 for each in df.gender]
df.group_alone = [1 if each == 'G' else 0 for each in df.group_alone]

# department column

df.loc[df['department'] == "Aeronautical Engineering", 'department'] = 0
df.loc[df['department'] == "Electronics and Automation Engineering", 'department'] = 1
df.loc[df['department'] == "Automotive Engineering", 'department'] = 2
df.loc[df['department'] == "Electrical Electronics Engineering", 'department'] = 3
df.loc[df['department'] ==
       "Electronics and Communication Engineering", 'department'] = 4
df.loc[df['department'] == "Mechanical Engineering", 'department'] = 5
df.loc[df['department'] == "Mechatronics Engineering", 'department'] = 6
df.loc[df['department'] == "Control and Automation Engineering", 'department'] = 7

# language column

df.loc[df['languages'] == "E", 'languages'] = 0
df.loc[df['languages'] == "G", 'languages'] = 1
df.loc[df['languages'] == "E,G", 'languages'] = 2
df.loc[df['languages'] == "Other", 'languages'] = 3
df.loc[df['languages'] == " ", 'languages'] = 4

# department_applied column

df.loc[df['department_applied'] == "Electronics", 'department_applied'] = 0
df.loc[df['department_applied'] == "Automation", 'department_applied'] = 1
df.loc[df['department_applied'] == "Programming", 'department_applied'] = 2
df.loc[df['department_applied'] == "Design", 'department_applied'] = 3
df.loc[df['department_applied'] == "Test", 'department_applied'] = 4

# military_sercive column

df.loc[df['military_service'] == "Exempted", 'military_service'] = 0
df.loc[df['military_service'] == "Delayed", 'military_service'] = 1
df.loc[df['military_service'] == "Remnant", 'military_service'] = 2
df.loc[df['military_service'] == "Uncertain", 'military_service'] = 3
df.loc[df['military_service'] == "Not at Military Age", 'military_service'] = 4
df.loc[df['military_service'] == " ", 'military_service'] = 5
df.loc[df['military_service'] == np.nan, 'military_service'] = 5

# internship column
df.loc[df['internship'] == "0", 'internship'] = 0
df.loc[df['internship'] == "1", 'internship'] = 1
df.loc[df['internship'] == " ", 'internship'] = 2


# degree column

df.loc[df['degree'] == "Bachelor", 'degree'] = 0
df.loc[df['degree'] == "Master", 'degree'] = 1
df.loc[df['degree'] == "PhD or Higher", 'degree'] = 2

# finally, hired not-hired values

df.hired = [1 if each == 'H' else 0 for each in df.hired]

# new unique values

for col in df:
    print(df[col].unique())

# checking if any NaN values left

sum = 0
for col in df:
    sum += df[col].isnull().sum()
print("Number of NaN values: " + str(sum))
# %%
# The correlation between columns

correlation = df.corr()
fig, ax = plt.subplots(figsize=(24, 24))
sns.heatmap(correlation, annot=True, linewidths=0.5, ax=ax)
plt.show()
# %%

corr_map_list = ["gpa", "reference", "age", "hired", "internship"]

sns.heatmap(df[corr_map_list].corr(), annot=True, fmt=".2f")
plt.show()
# %%

f, axes = plt.subplots(1, 2, figsize=(12, 8))

sns.countplot(x="hired", data=df, ax=axes[0])
sns.countplot(x="hired", hue='gender', data=df, ax=axes[1])
plt.show()
# %%

g = sns.FacetGrid(df, col="hired", height=7)
g.map(sns.distplot, "age", bins=25)
plt.show()
# %%

g = sns.FacetGrid(df, col="hired", height=7)
g.map(sns.distplot, "gpa", bins=25)
plt.show()
# %%

g = sns.FacetGrid(df, col="hired", height=7)
g.map(sns.distplot, "experience", bins=25)
plt.show()
# %%

g = sns.FacetGrid(df, col="hired", height=7)
g.map(sns.distplot, "department_applied", bins=4)
plt.show()
# %%
plt.figure(figsize=(25, 13))
ax = sns.barplot(y=df["department"].value_counts().values,
                 x=df["department"].value_counts().index, palette="ch:.25")
plt.title("Departments", color="black")
# %%
plt.figure(figsize=(25, 13))
ax = sns.barplot(y=df["languages"].value_counts().values,
                 x=df["languages"].value_counts().index, palette="ch:.25")
plt.title("Spoken Languages", color="black")
# %%
plt.figure(figsize=(25, 13))
ax = sns.barplot(y=df["military_service"].value_counts(
).values, x=df["military_service"].value_counts().index, palette="ch:.25")
plt.title("Military Service", color="black")
# %%
plt.figure(figsize=(25, 13))
ax = sns.barplot(y=df["department_applied"].value_counts(
).values, x=df["department_applied"].value_counts().index, palette="ch:.25")
plt.title("Applied Department", color="black")
# %%

# Splitted the dataset into 2 parts:
# y, the outputs
# x_, all of the dataset except the outputs.

y = df.hired.values
x_ = df.drop(columns=["hired"])
# %%

# normalising x values

x = (x_ - np.min(x_) / np.max(x_) - np.min(x_))
# %%

# spliting the dataset into train and test parts.

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1)

# in order to compare the scores, we created two lists
# accuracy and f1 score lists
# and a dictionary, so that we can graph it

accuracy_scores = {}
acc_scores = []
f1 = []
# %%

# SVM model

svm = SVC(random_state=1)
svm.fit(x_train, y_train)

# appending scores to the lists
acc_scores.append(float("{:.2f}".format(svm.score(x_test, y_test) * 100)))
accuracy_scores['SVM Score'] = (
    float("{:.2f}".format(svm.score(x_test, y_test) * 100)))

# prediction is necessary for f1 score and confusion matrix
y_pred_svm = svm.predict(x_test)
y_true_svm = y_test

f1.append(float("{:.2f}".format((f1_score(y_true_svm, y_pred_svm))*100)))


# Confusion Matrix
cm_svm = confusion_matrix(y_true_svm, y_pred_svm)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm_svm, annot=True, linewidths=0.5,
            linecolor="pink", fmt=".0f", ax=ax)
plt.xlabel(y_pred_svm)
plt.ylabel(y_true_svm)
plt.show()
# %%

# KNN model

knn_scores = []
for each in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=each)
    knn.fit(x_train, y_train)
    knn_scores.append(knn.score(x_test, y_test))

# Graph of the accuracy_scores

plt.plot(range(1, 20), knn_scores)
plt.xlabel("K values")
plt.ylabel("Accuracy")
plt.show()

accuracy_scores['KNN Score'] = (
    float("{:.2f}".format(knn.score(x_test, y_test) * 100)))
# %%

# Naive Bayes model

nb = GaussianNB()
nb.fit(x_train, y_train)

# appending scores to the lists
acc_scores.append(float("{:.2f}".format(nb.score(x_test, y_test) * 100)))
accuracy_scores['NB Score'] = (
    float("{:.2f}".format(nb.score(x_test, y_test) * 100)))


# prediction is necessary for f1 score and confusion matrix
y_pred_nb = nb.predict(x_test)
y_true_nb = y_test

f1.append(float("{:.2f}".format((f1_score(y_true_nb, y_pred_nb))*100)))


# Confusion Matrix
cm_nb = confusion_matrix(y_true_nb, y_pred_nb)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm_nb, annot=True, linewidths=0.5,
            linecolor="pink", fmt=".0f", ax=ax)
plt.xlabel(y_pred_nb)
plt.ylabel(y_true_nb)
plt.show()
# %%

# Decision Tree model

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

acc_scores.append(float("{:.2f}".format(dt.score(x_test, y_test) * 100)))
accuracy_scores['DT Score'] = (
    float("{:.2f}".format(dt.score(x_test, y_test) * 100)))

y_pred_dt = dt.predict(x_test)
y_true_dt = y_test

f1.append(float("{:.2f}".format((f1_score(y_true_dt, y_pred_dt))*100)))

cm_dt = confusion_matrix(y_true_dt, y_pred_dt)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm_dt, annot=True, linewidths=0.5,
            linecolor="pink", fmt=".0f", ax=ax)
plt.xlabel(y_pred_dt)
plt.ylabel(y_true_dt)
plt.show()
# %%

# Random Forest model

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(x_train, y_train)

acc_scores.append(float("{:.2f}".format(rf.score(x_test, y_test) * 100)))
accuracy_scores['RF Score'] = (
    float("{:.2f}".format(rf.score(x_test, y_test) * 100)))

y_pred_rf = rf.predict(x_test)
y_true_rf = y_test

f1.append(float("{:.2f}".format((f1_score(y_true_rf, y_pred_rf))*100)))

cm_rf = confusion_matrix(y_true_rf, y_pred_rf)
f, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(cm_rf, annot=True, linewidths=0.5,
            linecolor="pink", fmt=".0f", ax=ax)
plt.xlabel(y_pred_rf)
plt.ylabel(y_true_rf)
plt.show()
# %%
print(acc_scores)
print(f1)
# %%
lists = sorted(accuracy_scores.items())

x_axis, y_axis = zip(*lists)

plt.figure(figsize=(15, 10))
plt.plot(x_axis, y_axis)
plt.show()
#%%