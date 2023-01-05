# AI Support for HR Managers

## Table of Contents
- [Explanation of the Dataset](#explanation-of-the-dataset)
- [Requirements](#requirements)
- [Explanation of the Code](#explanation-of-the-code)

## Explanation of the Dataset

The dataset consists of 1101 rows, including the names of the columns, and 24 columns. These columns are:

- name: This columns consists of names of the applicants. It has 1033 unique values.
- surname: This columns consists of surnames of the applicants.
- mail: This columns consists of mails of the applicants.
- gender: This columns consists of genders of the applicants. It has 2 unique values, male and female.
- age: This columns consists of ages of the applicants. The age range was set to 18-45. It has 28 unique values.
- school: This columns consists of universities that applicants were graduated from. This column is constructed by using a list of all the universities in Turkey. Some universities that are considered good were doubled or tripled so that it would imitate the real hiring values. It has 160 unique values.
- department: This columns consists of departments that applicants were graduated from. This column is constructed by using a list of 8 different departments.
- city_living: This columns consists of the cities applicants are currently living. This column is constructed by using a list of all the cities in Turkey. The number of some big, crowded cities doubled or tripled so that it would imitate the real values. It has 62 unique values.
- experience: This columns consists of experiences of the applicants. The random values were assigned according to the "age" column. Excel IF formula is used for this column. It has 26 unique values.
- programs: This columns consists of the programs the applicants know how to use. This column is constructed by using a list of engineering programs, some rows had more than one program. It has 79 unique values.
- languages: This columns consists of the languages the applicants speak. It has 5 unique values: E (English), G (German), E,G (English and German), Other and None.
- gpa: This columns consists of GPAs of the applicants. The range was set to 2.0-4.0 in float form. It has 199 unique values.
- degree: This columns consists of the last degree the applicants got. It has 3 unique values: Bachelor, Master, PhD or Higher.
- department_applied: This columns consists of the department which the applicants applied for. It has 4 unique values: Electronics, Automation, Programming, Deisgn
- salary: This columns consists of the salary expectation of the applicants. This column is constructed by using random values between 10000-50000. All the values can be divided by 100. It has 43 unique values.
- drivers_licence: This columns consists of the information of whether the applicants have a drivers license or not. It has 2 unique values.
- smoking: This columns consists of the information of whether the applicants have a drivers license or not. It has 2 unique values.
- military_service: This columns consists of the military service information of the applicants. It has 6 unique values.
- flexible_hours: This columns consists of the information of whether the applicants are okay with flexible	working hours or not. It has 2 unique values.
- group_alone: This columns consists of the information of whether the applicants prefer working alone or with a group. It has 2 unique values.
- online: This columns consists of the information of whether the applicants are okay with working online or not. It has 2 unique values.
- reference: This columns consists of the information of whether the applicants have a reference or not. It has 2 unique values.
- internship: This columns consists of the information of whether the applicants have served their internships in the said company or not. It has 2 unique values.
- hired: This columns consists of the information of whether the applicants were hired or rejected. It has 2 unique values.

## Requirements

In our project, we used the following modules:

- From Sklearn:
  - train_test_split (model_selection)
  - DecisionTreeClassifier (tree)
  - SVC (svm)
  - GaussianNB (naive_bayes)
  - RandomForestClassifier (ensemble)
  - KNeighborsClassifier (neighbors)
  - confusion_matrix (metrics)
  - f1_score (metrics)
- Pandas
- Numpy
- Seaborn
- Matplotlib.pyplot

**The .csv file and .py file must be in the same directory. If the .csv file cannot be read, copy and paste .csv	file's path.**

## Explanation of the Code

### SECTION 1 - READING DATASET

The dataset is read and first three columns are dropped.

### SECTION 2 - DATASET ADJUSTMENTS

The comments explain the processes. We needed to getrid of the string values.

### SECTION 3 - VISUALIZATION

The data is visualized in this section. Several columns are compared to see the correlation between them.

### SECTION 4 - PREPARING TEST AND TRAIN SETS

The dataset is split into parts.

### SECTION 5 - CREATING MODELS AND TRAINING

The models are created in this section. The scores are appended to lists. Confusion matrices are plotted.



