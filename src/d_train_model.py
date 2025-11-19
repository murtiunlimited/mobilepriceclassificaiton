from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def train_svm(X_train, y_train):
    pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=42))
    ])

    param_grid = {
        'clf__C': [0.1, 1, 10, 100],
        'clf__kernel': ['linear', 'rbf', 'poly'],
        'clf__gamma': ['scale', 'auto']
    }

    grid = GridSearchCV(
        pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_
