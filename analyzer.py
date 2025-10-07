
import helper
import analysis_helper
import pandas as pd


class analyzer():
    def __init__(self):
        pass

    #Searching through dwt to find best results
    def dwt_search(self, manager, gsr_filtered, ecg_features, y_responses, emotion_responses):
        # Exploration of GSR with DWT starts here
        levels = [1, 2, 3, 4, 5]
        dbs = ['db2', 'db5']

        all_labels = []
        for level in levels:
            all_labels.append(helper.label_generator(level, True))

        # Exploring how to get highest accuracy using GSR DWT different wavelets and levels
        results = helper.transform_search_results_builder()
        for db in dbs:
            for label, level in zip(all_labels, levels):
                gsr_features = manager.gsr_feature_extraction(gsr_filtered, labels=label, db=db, level=level)
                # gsr_features.to_csv('./data/gsr_features.csv', index=False)
                # Fusing the data. Both Early and Late
                _, gsr_va, full_va = manager.data_fusion(ecg_features, gsr_features, y_responses)
                _, gsr_emo, full_emo = manager.data_fusion(ecg_features, gsr_features, emotion_responses)

                #Early Fusion
                early_fusion_data = helper.x_y_scale_split_bulk(full_va, full_emo)

                #Late Fusion
                late_fusion_data = helper.x_y_scale_split_bulk(gsr_va, gsr_emo)

                results = helper.transform_search('early', results, early_fusion_data, db, level)
                results = helper.transform_search('late', results, late_fusion_data, db, level)

        #Found accuracy for every stage, every category for every db and level. Need to find best wavelet and level for each category and stage
        results_df = pd.DataFrame(results)
        results_df.to_csv('./data/GSR DB exploration.csv')
        return helper.transform_best_results(results)

    #searching through dwt for ecg for best results
    def dwt_ecg_search(self, manager, best_gsr_transform_results, ecg_filtered, gsr_filtered, y_responses, emotion_responses):
        levels = [1, 2, 3, 4, 5]
        dbs = ['db2', 'db5', 'FFT']
        category = ['valence', 'arousal', 'full', 'emotion']

        all_labels = []
        for level in levels:
            all_labels.append(helper.label_generator(level, False))

        fft_labels = helper.fft_label_generator()

        #Pulling parameters and such using db and level from prior search.
        early_gsr, late_gsr = helper.get_best_results(best_gsr_transform_results)
        

        #Extracting GSR Features for use with early hand late fusion based on previous best results.
        #Early GSR Features
        early_gsr_features = []
        for i in range(4):
            db = early_gsr[i][0]
            level = early_gsr[i][1]
            features = manager.gsr_feature_extraction(gsr_filtered, labels=helper.label_generator(level, True), db=db, level=level)
            early_gsr_features.append(features)
        

        #Late GSR Features
        late_gsr_features = []
        for i in range(4):
            db = late_gsr[i][0]
            level = late_gsr[i][1]
            late_gsr_features.append(manager.gsr_feature_extraction(gsr_filtered, labels=helper.label_generator(level, True), db=db, level=level))
        
        ##Building a results dict similar to GSR results (early/late -> category -> results)
        results = helper.transform_search_results_builder()

        #Finding best transform and level for ECG
        for db in dbs:
            for label, level in zip(all_labels, levels):
                ##Extracting Features for ECG
                if db == 'FFT':
                    ecg_features = manager.ecg_feature_extraction(ecg_filtered, db, level, fft_labels, plot=False, FFT=True)
                else:
                    ecg_features = manager.ecg_feature_extraction(ecg_filtered, labels=label, db=db, level=level)

                ##Fusing gsr features and ECG features
                
                early_fusion = []
                late_fusion = []
                for ef, lf, cat, in zip(early_gsr_features, late_gsr_features, category):
                    if cat == 'emotion':
                        _,_,early = manager.data_fusion(ecg_features, ef, emotion_responses)
                        late,_,_ = manager.data_fusion(ecg_features, lf, emotion_responses)
                    else:
                        _,_, early = manager.data_fusion(ecg_features, ef, y_responses)
                        late,_,_ = manager.data_fusion(ecg_features, lf, y_responses)
                    early_fusion.append(early)
                    late_fusion.append(late)     
                        
                    

                ##Scaling and splitting the data
                ###Early Fusion
                early_fusion_data = []
                for data, cat in zip(early_fusion, category):
                    emo = False
                    if cat == 'emotion':
                        emo = True
                    early_fusion_data.append(helper.x_y_scale_split(data, emo=emo, target=cat))
                    

                ###Late Fusion
                late_fusion_data = []
                for data, cat in zip(early_fusion, category):
                    emo = False
                    if cat == 'emotion':
                        emo = True
                    late_fusion_data.append(helper.x_y_scale_split(data, emo=emo, target=cat))

                ##Searching for best results # BUG REPORTING = RF CLASSIFIER IS GETTING FLOATS AS THE Y. NEED TO EVALUATE THAT

                #FIRST ATTEMPTS TO EVALUATE
                results = helper.transform_search('early', results, early_fusion_data, db, level)
                results = helper.transform_search('late', results, late_fusion_data, db, level)

        results_df = pd.DataFrame(results)
        results_df.to_csv('./data/ECG DB exploration.csv')
        return helper.transform_best_results(results)

    def final_analysis(self, manager, ecg_filtered, gsr_filtered, best_gsr, best_ecg, y_responses, emotion_responses, verbose=False):
        early_data, late_ecg_data, late_gsr_data = analysis_helper.final_data_prep(manager, ecg_filtered, gsr_filtered, best_gsr, best_ecg, y_responses, emotion_responses)

        # True Values
        true_va = early_data[0][3]
        true_aro = early_data[1][3]
        true_full = early_data[2][3]
        true_emo = early_data[3][3]

        #Models
        models, params = analysis_helper.model_and_params()


        early_fusion_results = []
        guess_results = []
        single_signal_results = []
        multi_model_guesses = []
        late_fusion_labels = ['valence', 'arousal', 'full', 'emotion', 'combining_valence_arousal']
        multi_model_results = []
        
        #Early Fusion
        for model, param in zip(models, params):
            early_model_results = analysis_helper.model_search(early_data[0], early_data[1], early_data[2], early_data[3], model, param)
            ecg_results, gsr_results = analysis_helper.late_fusion_cv_search(late_ecg_data, late_gsr_data, model, param)

            early_fusion_results.append(early_model_results)
            single_signal_results.append(ecg_results)
            single_signal_results.append(gsr_results)

        early_df = pd.DataFrame(early_fusion_results)
        early_df.to_csv('./data/Early_Fusion.csv')

        late_df = pd.DataFrame(single_signal_results)
        late_df.to_csv('./data/Single Signal exploration.csv')
        

        #Late Fusion Below Here
        for model, param in zip(models, params):
            valence_predictions = []
            arousal_predictions = []
            full_predictions = []
            emotion_predictions = []
            vote_categories=[valence_predictions, arousal_predictions, full_predictions, emotion_predictions]

            #For each model, run it train and have it predict 5 times. Each prediction is a vote.
            for ecg, gsr, category, in zip(late_ecg_data, late_gsr_data, vote_categories):
                category.append(analysis_helper.voting(ecg, model, param))
                category.append(analysis_helper.voting(gsr, model, param))

            #Predicting the response by using majority vote 
            valence_guess = helper.majority_vote(valence_predictions)
            arousal_guess = helper.majority_vote(arousal_predictions)
            full_guess = helper.majority_vote(full_predictions)
            emotion_guess = helper.majority_vote(emotion_predictions)
            #Val_aro_fusion to determine if modles trained separately on valence and arousal, when combined, do well.
            val_aro_guess = [a + b for a, b in zip(valence_guess, arousal_guess)]

            all_guesses = [valence_guess, arousal_guess, full_guess, emotion_guess, val_aro_guess]
            #Append all guesses for the model to a larger list. larger list will be used to determine if votes across model types improves accuracy
            multi_model_guesses.append(all_guesses)

            #Determining Accuracy of models voting amongst themselves.
            all_answers = [true_va, true_aro, true_full, true_emo, true_full]
            for index, guess in enumerate(all_guesses):
                pct_correct = helper.accuracy_score(guess, all_answers[index])
                results = {
                    'Model': model,
                    'Target': late_fusion_labels[index],
                    'Accuracy': pct_correct
                }
                guess_results.append(results)

        guess_df = pd.DataFrame(guess_results)
        guess_df.to_csv('./data/Single Model Late Fusion Voting.csv')

        #Determining accuracy when voting takes place across all models instead of within models (each vote comes from different model)
        for i in range(5):
            multi_model_prediction = helper.majority_vote([guess[i] for guess in multi_model_guesses])
            results = {
                'Target': late_fusion_labels[i],
                'Accuracy': helper.accuracy_score(multi_model_prediction, all_answers[i])
            }
            multi_model_results.append(results)

        multi_model_df = pd.DataFrame(multi_model_results)
        multi_model_df.to_csv('./data/Multi Model Late Fusion Voting.csv')
        if verbose:
            analysis_helper.best_results(single_signal_results, early_fusion_results, guess_results, multi_model_results)
        return