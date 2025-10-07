import numpy as np
import helper
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import SelectKBest, f_classif

##Helper functions to analyzer

#Generating the models and parameters for this experiment
def model_and_params():
    # MLP
    mlp_structures = [(50, 2), (100,), (100, 2), (50,), (50, 20, 2)]
    mlp_alpha = np.arange(0.0001, 0.001, 0.0002)
    mlp_activators = ['tanh', 'relu']
    mlp_solvers = ['adam', 'sgd']
    mlp_learning_rate = np.arange(0.2, 1, 0.1)

    mlp_params = {'model__hidden_layer_sizes': mlp_structures, 'model__alpha': mlp_alpha,
                  'model__learning_rate_init': mlp_learning_rate,
                  'model__activation': mlp_activators,
                  'model__solver': mlp_solvers}
    mlp = MLPClassifier(max_iter=1000)

    # Random Forest Classifier
    tree_counts = list(range(10, 105, 5))
    leaf_size = list(range(1, 6, 1))
    tree_params = {'model__n_estimators': tree_counts,
                   'model__min_samples_leaf': leaf_size,
                   'model__bootstrap': [True, False]}
    rf = RandomForestClassifier()

    # Log Reg
    lr = LogisticRegression(max_iter=1000)
    log_params = {'model__penalty': ['l2', 'elasticnet'],
                  'model__l1_ratio': [0.15],
                  'model__solver': ['saga'],
                  'model__tol': np.arange(0.5, 1.6, 0.1)}

    # Ridge Reg
    rr = RidgeClassifier(max_iter=1000)
    ridge_params = {'model__tol': np.arange(0.5, 1.6, 0.1)}

    # KNN
    neighbor_count = list(range(2, 12, 2))
    ideal_neighbor_params = {'model__n_neighbors': neighbor_count}
    knn = KNeighborsClassifier()

    # SVC
    kernel_options = ['linear', 'poly', 'rbf', 'sigmoid']
    ideal_svc_params = {'model__C': np.arange(0.5, 2, 0.5), 'model__kernel': kernel_options}
    svc = SVC(max_iter=1000)

    # ADABOOST RF
    estimator_counts = list(range(10, 150, 10))
    ada_params = {'model__n_estimators': estimator_counts}
    ada = AdaBoostClassifier(estimator=RandomForestClassifier(), algorithm='SAMME')

    models = [mlp, rf, lr, rr, knn, svc, ada]
    params = [mlp_params, tree_params, log_params, ridge_params, ideal_neighbor_params, ideal_svc_params, ada_params]

    #Trimmed to fewer models
    #models = [rf, lr, knn]
    #params = [tree_params, log_params, ideal_neighbor_params]
    pipelines = [create_pipeline(model) for model in models]
    
    
    return pipelines, params

#Creating a pipeline to tune model by selecting best features for each model.
def create_pipeline(model):
    fea_selec = SelectKBest(f_classif, k=10)
    pipeline = Pipeline([('feature_selection', fea_selec), ('model', model)])
    return pipeline


#Given data and a model get the best accuracy and parameters for the 4 categories
def model_search(valence_data, arousal_data, full_data, emo_data, model, params):
    val_best, val_accuracy = model_run(*valence_data, model, params)
    aro_best, aro_accuracy = model_run(*arousal_data, model, params)
    full_best, full_accuracy = model_run(*full_data, model, params)
    emo_best, emo_accuracy = model_run(*emo_data, model, params)

    results = {
        'Model': model,
        'Valence_Accuracy': val_accuracy,
        'Valence_Parameters': val_best,
        'Arousal_Accuracy': aro_accuracy,
        'Arousal_Parameters': aro_best,
        'Full_Accuracy': full_accuracy,
        'Full_Parameters': full_best,
        'Emotion_Accuracy': emo_accuracy,
        'Emotion_Parameters': emo_best
    }
    return results

#Conducting a search with late fusion for best accuracy and parameters.
def late_fusion_cv_search(ecg_data, gsr_data, model, params):

    best_ecg_val, accuracy_ecg_val = model_run(*ecg_data[0], model, params)
    best_ecg_aro, accuracy_ecg_aro = model_run(*ecg_data[1],  model, params)
    best_ecg_full, accuracy_ecg_full = model_run(*ecg_data[2], model, params)
    best_ecg_emo, accuracy_ecg_emo = model_run(*ecg_data[3], model, params)

    best_gsr_val, accuracy_gsr_val = model_run(*gsr_data[0], model, params)
    best_gsr_aro, accuracy_gsr_aro = model_run(*gsr_data[1],  model, params)
    best_gsr_full, accuracy_gsr_full = model_run(*gsr_data[2], model, params)
    best_gsr_emo, accuracy_gsr_emo = model_run(*gsr_data[3], model, params)

    ecg_results = {
               'Signal': 'ECG',
               'Model': model,
               'Valence_Accuracy': accuracy_ecg_val,
               'Valence_Parameters': best_ecg_val,
               'Arousal_Accuracy': accuracy_ecg_aro ,
               'Arousal_Parameters': best_ecg_aro,
               'Full_Accuracy': accuracy_ecg_full,
               'Full_Parameters': best_ecg_full,
               'Emotion_Accuracy': accuracy_ecg_emo,
               'Emotion_Parameters': best_ecg_emo,
                }
    gsr_results = {
               'Signal': 'GSR',
               'Model': model,
               'Valence_Accuracy': accuracy_gsr_val,
               'Valence_Parameters': best_gsr_val,
               'Arousal_Accuracy': accuracy_gsr_aro,
               'Arousal_Parameters': best_gsr_aro,
               'Full_Accuracy': accuracy_gsr_full,
               'Full_Parameters': best_gsr_full,
               'Emotion_Accuracy': accuracy_gsr_emo,
               'Emotion_Parameters': best_gsr_emo,
               }
    return ecg_results, gsr_results

#Given x and y data run model and get accuracy.
def model_run(x_train, x_test, y_train, y_test, estimator, parameter):
    grid = GridSearchCV(estimator=estimator, param_grid=parameter, scoring='accuracy')
    grid.fit(x_train, y_train)
    best = grid.best_params_
    accuracy = grid.score(x_test, y_test)
    return best, accuracy

#Given data, getting predictions, needed for voting in later fusion.
def pred_model_run(x_train, x_test, y_train, estimator, param):
    gs = GridSearchCV(estimator=estimator, param_grid = param, scoring='accuracy')
    gs.fit(x_train, y_train)
    pred = gs.predict(x_test)
    return pred

#Given data get predictions from 5 models
def voting(data, estimator, param):
    votes = []
    data = data[:-1]
    for i in range(5):
        #Run a set of 5 independent models to cast "votes" on the correct responses
        votes.append(pred_model_run(*data, estimator, param))
    return votes

#Fusing valence and arousal  for late fusion
def pred_fuse(valence, arousal):
    return [a + b for a, b in zip(valence, arousal)]

#Some data prep for the final analysis
def final_data_prep(manager, ecg_filtered, gsr_filtered, best_gsr, best_ecg, y_responses, emotion_responses):
    #levels found by previous analysis and should be changed if your analysis yields different results
    early_gsr, late_gsr = helper.get_best_results(best_gsr)
    early_ecg, late_ecg = helper.get_best_results(best_ecg)

    #GSR Features
    early_gsr_features = manager.analysis_feature_extraction(gsr_filtered, early_gsr)
    late_gsr_features = manager.analysis_feature_extraction(gsr_filtered, late_gsr)

    #ECG Features
    early_ecg_features = manager.analysis_feature_extraction(ecg_filtered, early_ecg, gsr=False)
    late_ecg_features = manager.analysis_feature_extraction(ecg_filtered, late_ecg, gsr=False)

    # Since dealing with two different tables, need index in order to split them appropriately
    selected, unselected = helper.random_choice(gsr_filtered)
    is_emotion = [False, False, False, True]
    labels = ['valence', 'arousal', 'full', 'emotion']
    responses = [y_responses, y_responses, y_responses, emotion_responses]

    early_data = manager.fusion_scaling_splitting(early_gsr_features, early_ecg_features, responses, is_emotion, labels, selected, unselected)

    late_ecg_data, late_gsr_data = manager.fusion_scaling_splitting(late_gsr_features, late_ecg_features, responses, is_emotion, labels, selected, unselected, False)
    return early_data, late_ecg_data, late_gsr_data

#If verbose will print results for single signal, early fusion, late fusion, and ensemble voting
def best_results(control, ef, lf, ensemble):
    print('single', results_reporter(control, 'single'))
    print()
    print('EF', results_reporter(ef, 'early fusion'))
    print()
    print('LF', results_reporter(lf, 'single model late'))
    print()
    print('Voting', results_reporter(ensemble, 'ensemble'))
    return

#Finding the best results for each category based on the method (control, ef, lf, or ensemble voting)
def results_reporter(results, methodology):
    best_per = {
        "Valence": {"Accuracy": float('-inf'), "Model": None, "Parameters": None},
        "Arousal": {"Accuracy": float('-inf'), "Model": None, "Parameters": None},
        "Full": {"Accuracy": float('-inf'), "Model": None, "Parameters": None},
        "Emotion": {"Accuracy": float('-inf'), "Model": None, "Parameters": None},
    }
    for result in results:
        for cat in best_per.keys():
            
            if methodology == "single model late":
                best_per[cat]["Accuracy"] = result["Accuracy"]
                best_per[cat]["Model"] = result["Model"]
                best_per[cat]["Parameters"] = 'Single Model Late Fusion'
                
            elif methodology == 'ensemble':
                best_per[cat]["Accuracy"] = result["Accuracy"]
                best_per[cat]["Parameters"] = "Ensemble Late Fusion"
            
            else:
                x = f"{cat}_Accuracy"
                y = f"{cat}_Parameters"
                if methodology == 'single':
                    best_per[cat]["Signal"] = result['Signal']

                if result[x] > best_per[cat]['Accuracy']:
                    best_per[cat] = {
                        "Accuracy" : result[x],
                        "Model" : result['Model'],
                        "Parameters" : result[y]
                    }
    return best_per



