import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Reading the CSV file
df = pd.read_csv('adult_dataset.csv')

# Handling missing values
df = df[df['workclass'] != '?']
df = df[df['occupation'] != '?']
df = df[df['native.country'] != '?']

# Encoding categorical variables
df_categorical = df.select_dtypes(include=['object'])
le = preprocessing.LabelEncoder()
df_categorical = df_categorical.apply(le.fit_transform)
df = df.drop(df_categorical.columns, axis=1)
df = pd.concat([df, df_categorical], axis=1)
df['income'] = df['income'].astype('category')

# Data preparation
X = df.drop('income', axis=1)
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=99)

# Model building and evaluation
dt_default = DecisionTreeClassifier(max_depth=5)
dt_default.fit(X_train, y_train)
y_pred_default = dt_default.predict(X_test)

# Hyperparameter tuning
param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]
}

n_folds = 5
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dtree, param_grid=param_grid, cv=n_folds, verbose=1)
grid_search.fit(X_train, y_train)

# Best hyperparameters
print("Best accuracy:", grid_search.best_score_)
print(grid_search.best_estimator_)

# Model with optimal hyperparameters
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10, min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(X_train, y_train)

# Model evaluation
print(clf_gini.score(X_test, y_test))

# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred = clf_gini.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
