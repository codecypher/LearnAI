# Performance Metrics

```py
def evaluate(X_test, y_test):
    """
    Compute performance metrics and confusion matrix for classification problem. 
    """
    from sklearn.metrics import classification_report, f1_score, confusion_matrix, \
        accuracy_score, precision_score, recall_score
    
    predicted = model.predict(np.array([X_test[1]]))
    predicted_class = model.predict_classes(np.array([X_test[1]]))
    
    print('predicted:', predicted)
    print('predicted_class:', predicted_class)
    
    predicted_all = model.predict_classes(np.array(X_test))
    print('predicted_all:', predicted_all)
    
    # Have to convert one-hot encoding to actual classes: 0-9
    y_classes = ohe_to_classes(y_test)
    
    # string modulo operator
    #print("Accuracy: %.4f%%" % (accuracy_score(predicted_all, y_classes)*100))
    # using str.format()
    print('Accuracy:', '{0:.4f}%'.format(accuracy_score(predicted_all, y_classes)*100))
    #print('Precision', precision_score(predicted_all, y_classes))
    #print('Recall', recall_score(predicted_all, y_classes))
    #print('F1 Score', f1_score(predicted_all, y_classes))
    
    print("\nConufusion Matrix:\n", confusion_matrix(predicted_all, y_classes), "\n")
    print("Classification Report:\n", classification_report(predicted_all, y_classes))
```

```py
def compute_metrics(model, ds, scaler_pred):
    """
    Compute metrics on dataset
    
    TODO: Need to redo after changes to tf Datasets

    Args:
        ds (ndarray): dataset
        scaler_pred (MinMaxScaler): scaler
    """
    # Extract target values from dataset
    y = []
    for element in ds.as_numpy_iterator():
        X_elem, y_elem = element
        y = np.append(y, y_elem)
    y = y.reshape(len(y), 1)

    # Make predictions
    yhat = model.predict(ds)

    # Invert scaling for forecast
    # We concatenate the yhat column with the last n_features
    # of the dataset in order to inverse the scaling
    # inv_yhat = np.concatenate((yhat, X[:, -(n_features - 1):]), axis=1)
    inv_yhat = scaler_pred.inverse_transform(yhat)
    inv_yhat = inv_yhat[:, 0]  # (num_rows,)

    # Invert scaling for actual
    # inv_y = np.concatenate((y, X[:, -(n_features - 1):]), axis=1)
    inv_y = scaler_pred.inverse_transform(y)
    inv_y = inv_y[:, 0]  # (num_rows,)

    # print('\ninv_yhat:', inv_yhat.shape)
    # print('inv_y:', inv_y.shape)

    # Accuracy of predictions on dataset
    acc = 100 - (100 * (abs(inv_yhat - inv_y) / inv_y)).mean()

    mae = mean_absolute_error(inv_y, inv_yhat)
    mse = mean_squared_error(inv_y, inv_yhat)
    rmse = math.sqrt(mse)
    mape = mean_absolute_percentage_error(inv_y, inv_yhat)

    metrics_dict = {}
    metrics_dict["acc"] = acc
    metrics_dict["mae"] = mae
    metrics_dict["rmse"] = rmse
    metrics_dict["mape"] = mape

    # Standard Keras idiom for metrics (does not work here)
    # metric_names = [ 'loss', 'acc', 'mae', 'mse', 'mape' ]
    # result = model.evaluate(ds)
    #
    # # result_dict = dict(zip(model.metrics_names, result))
    # result_dict = dict(zip(metric_names, result))
    #
    # rmse = math.sqrt(result_dict['mse'])

    return metrics_dict
```

