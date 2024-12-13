```python
## Data Preprocessing

### **Importing Libraries**


import pandas as pd  

import numpy as np

import matplotlib.pyplot as plt

from pandas import DataFrame, Series

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression 

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from statsmodels.tools.tools import add_constant
```


```python
## Loading The Dataset
```


```python
df = pd.read_csv("C:/Users/THIS PC/Desktop/titanicdataset.csv")

print(df)
```

         PassengerId  Survived  Pclass  \
    0            892         0       3   
    1            893         1       3   
    2            894         0       2   
    3            895         0       3   
    4            896         1       3   
    ..           ...       ...     ...   
    413         1305         0       3   
    414         1306         1       1   
    415         1307         0       3   
    416         1308         0       3   
    417         1309         0       3   
    
                                                 Name     Sex   Age  SibSp  Parch  \
    0                                Kelly, Mr. James    male  34.5      0      0   
    1                Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   
    2                       Myles, Mr. Thomas Francis    male  62.0      0      0   
    3                                Wirz, Mr. Albert    male  27.0      0      0   
    4    Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1   
    ..                                            ...     ...   ...    ...    ...   
    413                            Spector, Mr. Woolf    male   NaN      0      0   
    414                  Oliva y Ocana, Dona. Fermina  female  39.0      0      0   
    415                  Saether, Mr. Simon Sivertsen    male  38.5      0      0   
    416                           Ware, Mr. Frederick    male   NaN      0      0   
    417                      Peter, Master. Michael J    male   NaN      1      1   
    
                     Ticket      Fare Cabin Embarked  
    0                330911    7.8292   NaN        Q  
    1                363272    7.0000   NaN        S  
    2                240276    9.6875   NaN        Q  
    3                315154    8.6625   NaN        S  
    4               3101298   12.2875   NaN        S  
    ..                  ...       ...   ...      ...  
    413           A.5. 3236    8.0500   NaN        S  
    414            PC 17758  108.9000  C105        C  
    415  SOTON/O.Q. 3101262    7.2500   NaN        S  
    416              359309    8.0500   NaN        S  
    417                2668   22.3583   NaN        C  
    
    [418 rows x 12 columns]
    


```python
## Data shape

print("The number for row:", df.shape[0])

print("The number for column:", df.shape[1])

```

    The number for row: 418
    The number for column: 12
    


```python
### Checking for data info


print("The data informations are:", df.info)
```

    The data informations are: <bound method DataFrame.info of      PassengerId  Survived  Pclass  \
    0            892         0       3   
    1            893         1       3   
    2            894         0       2   
    3            895         0       3   
    4            896         1       3   
    ..           ...       ...     ...   
    413         1305         0       3   
    414         1306         1       1   
    415         1307         0       3   
    416         1308         0       3   
    417         1309         0       3   
    
                                                 Name     Sex   Age  SibSp  Parch  \
    0                                Kelly, Mr. James    male  34.5      0      0   
    1                Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   
    2                       Myles, Mr. Thomas Francis    male  62.0      0      0   
    3                                Wirz, Mr. Albert    male  27.0      0      0   
    4    Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1   
    ..                                            ...     ...   ...    ...    ...   
    413                            Spector, Mr. Woolf    male   NaN      0      0   
    414                  Oliva y Ocana, Dona. Fermina  female  39.0      0      0   
    415                  Saether, Mr. Simon Sivertsen    male  38.5      0      0   
    416                           Ware, Mr. Frederick    male   NaN      0      0   
    417                      Peter, Master. Michael J    male   NaN      1      1   
    
                     Ticket      Fare Cabin Embarked  
    0                330911    7.8292   NaN        Q  
    1                363272    7.0000   NaN        S  
    2                240276    9.6875   NaN        Q  
    3                315154    8.6625   NaN        S  
    4               3101298   12.2875   NaN        S  
    ..                  ...       ...   ...      ...  
    413           A.5. 3236    8.0500   NaN        S  
    414            PC 17758  108.9000  C105        C  
    415  SOTON/O.Q. 3101262    7.2500   NaN        S  
    416              359309    8.0500   NaN        S  
    417                2668   22.3583   NaN        C  
    
    [418 rows x 12 columns]>
    


```python
### **Checking for missing values**
```


```python
missing_values = df.isnull().sum()

#### Print the missing values for each column

print(missing_values)

#### Check if there are any missing values in the entire dataset

if missing_values.sum() == 0:
    print("No missing values in the dataset.")
else:
    print("There are missing values in the dataset.")
```

    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age             86
    SibSp            0
    Parch            0
    Ticket           0
    Fare             1
    Cabin          327
    Embarked         0
    dtype: int64
    There are missing values in the dataset.
    


```python
### Dropping Cabin Column
```


```python
df.drop('Cabin', axis =1, inplace=True)

print(df.isna().sum())
```

    PassengerId     0
    Survived        0
    Pclass          0
    Name            0
    Sex             0
    Age            86
    SibSp           0
    Parch           0
    Ticket          0
    Fare            1
    Embarked        0
    dtype: int64
    


```python
#### Handling missing values in Age column

df['Age'].fillna(df['Age'].mean(), inplace=True)
```


```python
print(df.isna().sum())
```

    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           1
    Embarked       0
    dtype: int64
    


```python
df['Fare'].fillna(df['Fare'].median(), inplace=True)
```


```python
print(df.isna().sum())
```

    PassengerId    0
    Survived       0
    Pclass         0
    Name           0
    Sex            0
    Age            0
    SibSp          0
    Parch          0
    Ticket         0
    Fare           0
    Embarked       0
    dtype: int64
    


```python
## Exploratory Data Analysis (EDA)  

### Descriptive Statistics

print(df.describe().T)
```

                 count         mean         std     min       25%         50%  \
    PassengerId  418.0  1100.500000  120.810458  892.00  996.2500  1100.50000   
    Survived     418.0     0.363636    0.481622    0.00    0.0000     0.00000   
    Pclass       418.0     2.265550    0.841838    1.00    1.0000     3.00000   
    Age          418.0    30.272590   12.634534    0.17   23.0000    30.27259   
    SibSp        418.0     0.447368    0.896760    0.00    0.0000     0.00000   
    Parch        418.0     0.392344    0.981429    0.00    0.0000     0.00000   
    Fare         418.0    35.576535   55.850103    0.00    7.8958    14.45420   
    
                         75%        max  
    PassengerId  1204.750000  1309.0000  
    Survived        1.000000     1.0000  
    Pclass          3.000000     3.0000  
    Age            35.750000    76.0000  
    SibSp           1.000000     8.0000  
    Parch           0.000000     9.0000  
    Fare           31.471875   512.3292  
    


```python
print(df.columns)
```

    Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
           'Parch', 'Ticket', 'Fare', 'Embarked'],
          dtype='object')
    


```python
## visualizations
```


```python

categorical_variables = ['Pclass', 'Sex', 'Parch', 'Embarked', 'SibSp']

fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.flatten()

for i, categorical_variable in enumerate(categorical_variables):
    sns.countplot(x=categorical_variable, hue='Survived', data=df, ax=axes[i])
    axes[i].set_title(f"Distribution of {categorical_variable} by Survival", fontsize=12)
    axes[i].legend(title='Survived', fontsize=10, loc='upper right') 

plt.tight_layout()

fig.suptitle("Comparison of Categorical Variables with Survival", fontsize=14, y=1)
plt.show()

```


    
![png](output_16_0.png)
    



```python
categorical_variables = ['Pclass', 'Sex', 'Parch', 'Embarked', 'SibSp']

fig, axes = plt.subplots(2, 3, figsize=(18, 8))
axes = axes.flatten()
for i, categorical_variable in enumerate(categorical_variables):
    sns.boxplot(x=categorical_variable, y='Age', hue='Survived', data=df, ax=axes[i])
    axes[i].set_title(f"Distribution of Age by {categorical_variable} and Survival", fontsize=12)
    axes[i].legend(title='Survived', fontsize=10, loc='upper right')  

plt.tight_layout()

fig.suptitle("Comparison of Categorical Variables with Survival (Boxplot)", fontsize=14, y=1.02)

plt.show()

```


    
![png](output_17_0.png)
    



```python
numerical_data= df[['Age','Fare',]]
print(numerical_data)

fig, ax = plt.subplots(1,2, figsize= (10,6))
fig.suptitle("Numerical data visualisation")
for i,col in enumerate(numerical_data.columns):
    axs=ax[i%3]
    sns.histplot(x=df[col],ax=axs, kde=True)
    axs.set_title(col, fontsize=16)
    plt.tight_layout()
plt.show()
```

              Age      Fare
    0    34.50000    7.8292
    1    47.00000    7.0000
    2    62.00000    9.6875
    3    27.00000    8.6625
    4    22.00000   12.2875
    ..        ...       ...
    413  30.27259    8.0500
    414  39.00000  108.9000
    415  38.50000    7.2500
    416  30.27259    8.0500
    417  30.27259   22.3583
    
    [418 rows x 2 columns]
    


    
![png](output_18_1.png)
    



```python
### Model Building

####  Train-Test Split

X= df[['Age','Pclass','Fare']]
y= df['Survived']
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3,
                                                   random_state=42)


#### LOGISTIC REGRESSION
model= LogisticRegression(random_state=42)

#Train the model
model.fit(X_train, y_train)

#Make prediction
y_pred = model.predict(X_test)

#### Model Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
target_names =['Survived', 'Not survived']
print(metrics.confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred,target_names = target_names))

#Logistics Regression Analysis

X_train_sm = sm.add_constant(X_train)

#Fit the model
logit_model = sm.Logit(y_train, X_train_sm)
result = logit_model.fit()
print(result.summary())
```

    Accuracy: 0.6666666666666666
    [[80  5]
     [37  4]]
    Classification Report:
                   precision    recall  f1-score   support
    
        Survived       0.68      0.94      0.79        85
    Not survived       0.44      0.10      0.16        41
    
        accuracy                           0.67       126
       macro avg       0.56      0.52      0.48       126
    weighted avg       0.61      0.67      0.59       126
    
    Optimization terminated successfully.
             Current function value: 0.642592
             Iterations 5
                               Logit Regression Results                           
    ==============================================================================
    Dep. Variable:               Survived   No. Observations:                  292
    Model:                          Logit   Df Residuals:                      288
    Method:                           MLE   Df Model:                            3
    Date:                Fri, 13 Dec 2024   Pseudo R-squ.:                 0.03243
    Time:                        06:57:36   Log-Likelihood:                -187.64
    converged:                       True   LL-Null:                       -193.93
    Covariance Type:            nonrobust   LLR p-value:                  0.005641
    ==============================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         -0.0680      0.683     -0.100      0.921      -1.407       1.271
    Age           -0.0066      0.011     -0.618      0.536      -0.027       0.014
    Pclass        -0.1986      0.194     -1.022      0.307      -0.579       0.182
    Fare           0.0057      0.003      2.050      0.040       0.000       0.011
    ==============================================================================
    


```python

```
