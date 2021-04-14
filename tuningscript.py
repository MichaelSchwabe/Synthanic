from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

# Naiven
from sklearn.naive_bayes import GaussianNB

# linear
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression

# GaussianProcess
# https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


class StructuredClassifierTuning():
    
    def get_best_params_and_report(self, results, n_top=5):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0} Mean validation score: {1:.3f} (std: {2:.3f})".format(i,
                                                                                                  results['mean_test_score'][candidate],
                                                                                                  results['std_test_score'][candidate]))
                #print("Parameters: {0}".format(results['params'][candidate]))
                # print("")
                if i == 1:
                    bestparams = results['params'][candidate]
                    bestscore = {'mean': results['mean_test_score']
                                 [candidate], 'std': results['std_test_score'][candidate]}
        return bestparams, bestscore

    def best_opti_fitted_model(self, model_list, models_name_list, param_dict_list, x, y, xtest, ytest, verbose=0, pre_dispatch=None, n_jobs=None, folds=2, iter_search=5):
        xnew = x
        ynew = y
        xtest = xtest
        ytest = ytest

        # die ganzen score arrays ... koennen ausgeduent werden !!!!
        precision_array = []
        accuracy_array = []
        accuracyBalanced_array = []
        recall_array = []
        TPR_array = []
        TNR_array = []
        Fmeasure_array = []
        name_array = []
        time_to_optimized_hyperParameters = []
        # return Values für die rueckgabe
        bestACC = 0.0
        bestF1 = 0.0
        bestPre = 0.0
        bestRec = 0.0
        bestmodel = 0
        bestscoreList = []
        bestscoreList2 = []
        # RandomizedHyperparamter
        n_iter_search = iter_search

        for i in range(len(model_list)):
            # RandomInitializedModel
            random_search = RandomizedSearchCV(
                model_list[i], param_distributions=param_dict_list[i], n_iter=n_iter_search, cv=folds, verbose=verbose, pre_dispatch=pre_dispatch, n_jobs=n_jobs) #iid=False
            start = time()
            random_search.fit(xnew, ynew)
            time_to_optimized_hyperParameters.append(time() - start)
            print("\n######### " + str(models_name_list[i]) + " #########")
            print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings." % (
                time_to_optimized_hyperParameters[i], n_iter_search))
            
            bestparams, bestscore = self.get_best_params_and_report(random_search.cv_results_)
            
            bestscoreList.append(bestscore['mean'])
            bestscoreList2.append(bestscore['std'])
            model_list[i] = random_search.best_estimator_
            model_list[i].fit(xnew, ynew)

            # Berechnung der Scores
            precision_array.append(precision_score(
                ytest, model_list[i].predict(xtest)))
            accuracy_array.append(accuracy_score(
                ytest, model_list[i].predict(xtest)))
            recall_array.append(recall_score(ytest, model_list[i].predict(xtest)))
            TPR_array.append(recall_score(ytest, model_list[i].predict(xtest)))
            TNR_array.append(
                (1 - (recall_score(ytest, model_list[i].predict(xtest)))))
            Fmeasure_array.append(
                2 * ((precision_array[i] * recall_array[i]) / (precision_array[i] + recall_array[i])))
            accuracyBalanced_array.append((TNR_array[i] + TPR_array[i]) / 2)
            name_array.append(models_name_list[i])

            # Selektion Section
            tempACC = accuracy_score(ytest, model_list[i].predict(xtest))
            tempPre = precision_score(ytest, model_list[i].predict(xtest))
            tempF1 = 2 * ((precision_array[i] * recall_array[i]) /
                          (precision_array[i] + recall_array[i]))
            tempRec = recall_score(ytest, model_list[i].predict(xtest))
            print(tempACC)
            if bestF1 < tempF1 and bestPre < tempPre and bestRec < tempRec:
                # if bestACC < tempACC:
                print('########  Treffer -> Save -> ' +
                      str(name_array[i]) + ' ########')
                bestACC = tempACC
                bestF1 = tempF1
                bestPre = tempPre
                bestRec = tempRec
                bestIndex = i

                bestmodel = model_list[i]
        d = {'Model': name_array,
             'Precision': precision_array,
             'Accuracy': accuracy_array,
             'Balanced Accuracy': accuracyBalanced_array,
             'Recall': recall_array,
             'TPR': TPR_array,
             'TNR': TNR_array,
             'Fmeasure': Fmeasure_array,
             'Opti_MeanValidationScore': bestscoreList,
             'Opti_StdScore': bestscoreList2,
             'Opti_TimeToHyP': time_to_optimized_hyperParameters}
        return pd.DataFrame(data=d), [bestIndex, bestmodel, bestACC, bestPre, bestF1], model_list

    def get_my_models_list(self):    
        return [
            RandomForestClassifier(),
            BaggingClassifier(),
            ExtraTreesClassifier(),
            AdaBoostClassifier(),
            # GradientBoostingClassifier(),
            # VotingClassifier(estimators=1000),
            GaussianNB(),
            # RidgeClassifier(),
            LogisticRegression()
            ]
    def get_my_models_name_list(self):
        return [
            'RandomForestClassifier (Ensamble)',
            'BaggingClassifier (Ensamble)',
            'ExtraTreesClassifier (Ensamble)',
            'AdaBoostClassifier (Ensamble)',
            #'GradientBoostingClassifier (Ensamble)',
            'GaussianNB (NaiveBayes)',
            #'RidgeClassifier (linear)',
            'LogisticRegression (linear)'
            ]
        

    # DEFINE THE CPUS YOU WILL USE FOR TRAIN AND OPTIMAZATION


    # # ParameterGrid anlegen ... für jedes Modell bzw. Paramtervariationen abdecken
    # DEFINE THE PARAMETE
    def get_parameter_distribution_list(self, cores = 2):
        cpu_cores = cores
        return [{"max_depth": [1, 3, 5, 7, 9, 11, 15, 20, 35, None],
                   # "max_features": sp_randint(1, 11),
                   "min_samples_split": sp_randint(2, 15),
                   "bootstrap": [True, False],
                   "criterion": ["gini", "entropy"],
                   "class_weight": [None],
                   "max_leaf_nodes": [None],
                   "min_impurity_decrease": [0.0, 0.1, 0.2],
                   "min_impurity_split": [None],
                   "min_samples_leaf": sp_randint(1, 15),
                   "n_estimators": [300, 400, 600, 800, 1000, 2000, 4000],
                   "n_jobs": [cpu_cores],
                   "min_weight_fraction_leaf": [0.0, 0.1, 0.2],
                   # "random_state": [None, 42, 132],
                   "warm_start": [True, False]
                   },
                  {"bootstrap": [True, False],
                   "bootstrap_features": [True, False],
                   # "max_features": sp_randint(1, 11),
                   "max_samples": [1.0],
                   "n_estimators": [100, 200, 500, 600, 700, 800, 1000],
                   "n_jobs": [cpu_cores],
                   # "random_state": [None, 42, 132],
                   "warm_start": [True, False]
                   },
                  {"max_depth": [1, 3, 5, 7, 9, 11, None],
                   # "max_features": sp_randint(1, 11),
                   "min_samples_split": sp_randint(2, 11),
                   "bootstrap": [True, False],
                   "criterion": ["gini", "entropy"],
                   "class_weight": [None],
                   "max_leaf_nodes": [None],
                   "min_impurity_decrease": [0.0, 0.1, 0.2],
                   "min_impurity_split": [None],
                   "min_samples_leaf": sp_randint(1, 11),
                   "min_samples_split": sp_randint(2, 11),
                   "n_estimators": [200, 300, 400, 500, 600, 700, 800, 1000],
                   "n_jobs": [cpu_cores],
                   "min_weight_fraction_leaf": [0.0, 0.1, 0.2],
                   # "random_state": [None, 42, 132],
                   "warm_start": [True, False]
                   },
                  {"algorithm": ['SAMME.R', 'SAMME'],
                   "learning_rate": [1.0],
                   "n_estimators": [100, 200, 500, 600, 700, 800, 900, 1000],
                   "random_state": [None, 42, 132],
                   },
                  # {"criterion": ['friedman_mse', 'mse', 'mae'], #mean absolute error (mae).
                  # "learning_rate": [0.1],
                  # "loss": ['deviance', 'exponential'],
                  # "max_depth": sp_randint(3, 7),
                  # "min_samples_leaf": sp_randint(1, 11),
                  # "min_samples_split": sp_randint(2, 11),
                  # "n_estimators": [10,20,30],
                  # "min_weight_fraction_leaf": [0.0, 0.1, 0.2],
                  # "random_state": [42,],
                  # "warm_start": [True, False],
                  # "presort": ['auto'],
                  # "subsample": [1.0],
                  # "tol": [0.0001],
                  # "validation_fraction": [0.1]
                  # },
                  {"var_smoothing": [1e-09]
                   },
                  {"C": [1.0, 0.5, 1.1, 0.9],
                   "fit_intercept": [True, False],
                   "intercept_scaling": sp_randint(1, 11),
                   "l1_ratio": [None],
                   "max_iter": [100, 200, 500, 1000],
                   "multi_class": ['ovr', 'multinomial', 'auto'],
                   "n_jobs": [cpu_cores],
                   "penalty": ['l2'],
                   "tol": [0.0001],
                   # "random_state": [None, 42, 132],
                   "warm_start": [True, False],
                   "solver": ['lbfgs', 'sag', 'saga'],
                   }]
    
    def plot_roccurve(self, my_models,my_models_name, X_test, y_test):
        rf_roc_auc_array = []
        fpr_array = []
        tpr_array = []
        thresholds_array = []
        
        plt.figure(figsize=(17,10))
        for i in range(len(my_models)):
            rf_roc_auc_array.append(roc_auc_score(y_test, my_models[i].predict(X_test)))
            fpr, tpr, thresholds = roc_curve(y_test, my_models[i].predict_proba(X_test)[:,1])
            fpr_array.append(fpr)
            tpr_array.append(tpr)
            thresholds_array.append(thresholds)
            plt.plot(fpr, tpr, label=my_models_name[i]+' (area = %0.2f)' % rf_roc_auc_array[i])

        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate / Falsch erkannte Treffer')
        plt.ylabel('True Positive Rate / Richtig erkannte Treffer')
        plt.title('Receiver operating characteristic - ROC-Graph -> Optimierte Modelle')
        plt.legend(loc="lower right")
        plt.savefig('RF_ROC_opti3')
        plt.show()
        
    def save_best_models(self, my_models,my_models_name):
        for i in range(len(my_models)):
            print('SAVE MODEL ' + str(my_models_name[i]) + ' .. ')
            joblib.dump(my_models[i], str(my_models_name[i]) + "_HP_opti.joblib")
            print('SAVE COMPLETE !')