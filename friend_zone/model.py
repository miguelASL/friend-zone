from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


def train_model(X_train, y_train, feature_names):
    clf = RandomForestClassifier(random_state=42)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }

    grid_search = GridSearchCV(
        estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_clf = grid_search.best_estimator_
    best_clf.fit(X_train, y_train)

    print("Modelo entrenado exitosamente con los mejores parámetros encontrados.")
    best_clf.feature_names_in_ = feature_names
    return best_clf


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    print("Exactitud del modelo:", accuracy_score(y_test, y_pred))
