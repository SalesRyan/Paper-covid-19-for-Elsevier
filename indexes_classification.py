'''
This code performs the classification of the texture descriptors presented in the paper
"COVID-index: a texture-based approach to classify lung lesions based on CT images".

The features were extracted from the databases mentioned in the article, which have images of covid-19 lesions,
solid lesions and healthy tissue (non-nodules).

Classifiers: Random Forest e XGBoost.

Important notes:
    * Experiment 1: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 1) and COVID-19 (dataset 2)
    * Experiment 2: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 1)
    * Experiment 3: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 2)

Developed by: Patrick Sales and Vitória Carvalho.

Last update: April 2021.
'''
# -----------------------------------------------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer
import time
from glob import glob

# -----------------------------------------------------------------------------------------------------------------------------

def classification(X, y, k=5, classifier='rf'):

    '''
    This function performs feature classification through the Random Forest and XGBoost classifiers,
    using the cross-validation technique.

    Parameters:
        X: numpy array, required
            A numpy array with the features that will be classify.

        y: numpy array, required
            A one-dimensional numpy array that stores the feature labels.

        k: int, default 5
            Variable that defines the number of folds in the cross-validation, i.e. the number of splits that
            will be performed.

        classifier: str, default 'rf'
            Classifier that will be used.
                'rf': Random Forest Classifier
                'xgboost': XGBoost Classifier

    Returns:
        results: dict
            A dictionary containing the classification results.
    '''
    
    conf_matrix = np.zeros([3,3], dtype=int)
    matrices_list = []
    sumAcc = []
    sumRecall = []
    sumPrecision = []
    sumF1 = []
    sumAUC = []
    sumKappa = []
    
    classifier_model = {
        'rf': RandomForestClassifier(n_jobs=20),
        'xgboost': XGBClassifier(n_jobs=20)
    }
    
    print(f'\n<< Cross validation with {classifier} classifier >>\n')

    begin = time.time()

    model = classifier_model[classifier]
    cv = KFold(k, random_state=0, shuffle=True)
    count = 1

    for train_index, test_index in cv.split(X):

      #Split fold
      X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

      #Train and predict model
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      #Metrics
      sumAcc.append(accuracy_score(y_test, y_pred))
      sumRecall.append(recall_score(y_test, y_pred, average='weighted'))
      sumPrecision.append(precision_score(y_test, y_pred, average='weighted'))
      sumF1.append(f1_score(y_test, y_pred, average='weighted'))
      sumKappa.append(cohen_kappa_score(y_test, y_pred))
      cm = confusion_matrix(y_test, y_pred)
      conf_matrix += cm
      matrices_list.append(cm)

      lb = LabelBinarizer()
      lb.fit(y_test)
      true = lb.transform(y_test)
      pred = lb.transform(y_pred)

      sumAUC.append(roc_auc_score(true, pred))

      print(f'{count}/{k} done!')
      count += 1
    
    end = time.time()
    
    # Time, mean and std of results
    results = {
        'time': end-begin,
        'acc': f'{np.asarray(sumAcc).mean()} ± {np.asarray(sumAcc).std()}',
        'rec': f'{np.asarray(sumRecall).mean()} ± {np.asarray(sumRecall).std()}',
        'prec': f'{np.asarray(sumPrecision).mean()} ± {np.asarray(sumPrecision).std()}',
        'f1': f'{np.asarray(sumF1).mean()} ± {np.asarray(sumF1).std()}',
        'auc': f'{np.asarray(sumAUC).mean()} ± {np.asarray(sumAUC).std()}'
    }
    
    # Showing results
    print(f'\nTime: {end-begin}')
    print('Accuracy: {} (+/- {})'.format(np.asarray(sumAcc).mean(), np.asarray(sumAcc).std()))
    print('Recall: {} (+/- {})'.format(np.asarray(sumRecall).mean(), np.asarray(sumRecall).std()))
    print('Precision: {} (+/- {})'.format(np.asarray(sumPrecision).mean(), np.asarray(sumPrecision).std()))
    print('F1-score: {} (+/- {})'.format(np.asarray(sumF1).mean(), np.asarray(sumF1).std()))
    print('AUC: {} (+/- {})'.format(np.asarray(sumAUC).mean(), np.asarray(sumAUC).std()))
    print('Kappa: {} (+/- {})'.format(np.asarray(sumKappa).mean(), np.asarray(sumKappa).std()))
    
    return results
    
# -----------------------------------------------------------------------------------------------------------------------------

features_path = './indexes_features.csv'

for experiment in [1, 2, 3]:

    final_results = {
        'classifier': [],
        'time': [],
        'acc': [],
        'rec': [],
        'prec': [],
        'f1': [],
        'auc': []
    }

    print(f'\n<< Experiment {experiment} >>\n')

    for classifier_name in ['rf', 'xgboost']:
            
        # Loading DataFrame with the features
        df = pd.read_csv(features_path)

        # Experiment 1: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 1) and COVID-19 (dataset 2)
        if experiment == 1:
            df.loc[df.label > 2, 'label'] = 1

        # Experiment 2: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 1)
        elif experiment == 2:
            df.loc[df.label == 3, 'label'] = 1
            df = pd.concat([df[df.label == 0], df[df.label == 1], df[df.label == 2]], ignore_index=True)

        # Experiment 3: Healthy tissue (non-nodules) x solid x COVID-19 (dataset 2)
        elif experiment == 3:
            df.loc[df.label == 4, 'label'] = 1
            df = pd.concat([df[df.label == 0], df[df.label == 1], df[df.label == 2]], ignore_index=True)

        # Separating X and y in the data
        X = np.array(df.drop(['name', 'label'], axis=1))
        y = np.array(df.label)
        print(f'\nLabels: {np.unique(y)}, len(X) = {len(X)}, len(y) = {len(y)}\n')

        # Performing classification
        results =  classification(X, y, k=5, classifier=classifier_name)

        # Putting results in the dictionary
        final_results['classifier'].append(classifier_name)
        final_results['time'].append(results['time'])
        final_results['acc'].append(results['acc'])
        final_results['rec'].append(results['rec'])
        final_results['prec'].append(results['prec'])
        final_results['f1'].append(results['f1'])
        final_results['auc'].append(results['auc'])
            
    # Saving results of each experiment in a .csv file
    pd.DataFrame(final_results).to_csv(f'./experiment_{experiment}.csv', index=False)
        
print('\n\n<< DONE >>\n')
    
# -----------------------------------------------------------------------------------------------------------------------------
