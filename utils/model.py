from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score


def binary_label_encoding(X):
    return X.map(lambda x: 1 if x in ['Yes', 'Urban', True] else 0)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
