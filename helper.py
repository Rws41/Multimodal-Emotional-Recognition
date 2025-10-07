import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


#Adding descriptive columns to results
def db_results_adjuster(model_results, db, level, Fusion="Early"):
    model_results['wavelet'] = db
    model_results['level'] = level
    model_results['Fusion'] = Fusion
    return model_results


#Randomly breaking up data
def random_choice(data):
    # Since dealing with two different tables, need index in order to split them appropriately
    n = data.shape[0]
    samples = int(0.7 * n)
    selected = np.random.choice(np.arange(n), replace=False, size=samples)
    unselected = np.setdiff1d(np.arange(n), selected)
    return selected, unselected

#Given a list of predictions find which is the most common prediction for each model.
def majority_vote(predictions):
    df = pd.DataFrame(np.column_stack(predictions))
    most_common = df.mode(axis=1).iloc[:, 0]
    return most_common

#Calculating the percent accuracy
def accuracy_score(guess, true):
    # Comparing Results to true values
    x = 0
    n = len(true)
    for a, b in zip(guess, true):
        if a == b:
            x += 1
    return x/n

#splitting data into x and y
def data_split(data, emo, target):
    if emo == True:
        x = data.iloc[:, 0:-1]
    else:
        x = data.iloc[:, :-3]

    if target == 'full' or target == 'emotion':
        y = data.iloc[:, -1]
    elif target == 'arousal':
        y=data.iloc[:,-2]
    elif target == 'valence':
        y=data.iloc[:,-3]
    return x, y

#splitting into train and testing data - scaled by min max
def train_test_scale_split(data, selected, unselected, emo=False, target='valence'):
    x, y = data_split(data, emo, target)
    #Scaling it
    x_scaled = scaler(x)

    train_x = x_scaled[selected]
    test_x = x_scaled[unselected]

    train_y = y[selected]
    test_y = y[unselected]
    return train_x, test_x, train_y, test_y

def x_y_scale_split(data, emo=False, target= 'valence'):
    x, y = data_split(data, emo, target)
    x_scaled = scaler(x)
    return x_scaled, y

def x_y_scale_split_bulk(va_data, emo_data):
    valence_data = x_y_scale_split(va_data, False, 'valence')
    arousal_data = x_y_scale_split(va_data, False, 'arousal')
    full_data = x_y_scale_split(va_data, False, 'full')
    emo_data = x_y_scale_split(emo_data, True, 'emotion')
    return [valence_data, arousal_data, full_data, emo_data]

def scaler(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)

def label_generator(level, gsr=False):
    metrics = ['energy', 'mean', 'std_dev', 'max', 'min']
    if gsr:
        labels = ['Session', 'Participant', 'Video', 'gsr_average', 'gsr_standard_dev',
              'gsr_kurtosis', 'gsr_skew', 'gsr_variance', 'gsr_range',]
        for i in range(level+1):
            y = i+1
            for metric in metrics:
                labels.append(f'coef{y}_' + metric)
        labels.append('gsr_entropy')
        return labels
    else:
        labels = ['Session', 'Participant', 'Video', 'rr_mean', 'rr_std', 'rmssd', 'pnn50',]
        for i in range(level+1):
            y = i+1
            for metric in metrics:
                labels.append(f'coef{y}_' + metric)
        ending_labels = ['ecg_entropy', 'ecg_average', 'ecg_standard_dev',
              'ecg_kurtosis', 'ecg_skew', 'ecg_variance',]
        for label in ending_labels:
            labels.append(label)
        return labels
    
    
def fft_label_generator():
    labels = ['Session', 'Participant', 'Video', 'rr_mean', 'rr_std', 'rmssd', 'pnn50',
                  'ecg_spectral_centroid', 'ecg_power_ratio',
                  'ecg_low_freq_skew', 'ecg_high_freq_skew', 'ecg_lf_range', 'ecg_hf_range',
                  'ecg_frequency_kurtosis', 'ecg_frequency_skew', 'ecg_frequency_standard_dev', 'ecg_frequency_mean',
                  'ecg_average', 'ecg_standard_dev', 'ecg_kurtosis', 'ecg_skew', 'ecg_variance', ]
    return labels
def transform_cross_val_scoring(x, y):
    rf = RandomForestClassifier()
    scores = cross_val_score(rf, x, y, cv=5, scoring='accuracy')
    return np.mean(scores)

def transform_search_results_builder():
    results = {}
    stage = ['early', 'late']
    category = ['valence', 'arousal', 'full', 'emotion']
    for i in stage:
        results[i] = {}
        for j in category:
            results[i][j] = []
    return results

def transform_search(stage, results, data, db, level):
    cat_label = ['valence', 'arousal', 'full', 'emotion']
    for x, y in zip(cat_label, data):
        results[stage][x].append({
            'db': db,
            'level': level,
            'accuracy': transform_cross_val_scoring(*y)
        })
    return results
    
def transform_best_results(results):
    best_results = {}
    for stage in results:
        best_results[stage] = {}
        for category, records in results[stage].items():
            best_result = max(records, key=lambda r: r["accuracy"])
            best_results[stage][category] = best_result

    return best_results

def get_best_results(best_data):
    early = []
    late = []
    timing = ['early', 'late']
    category = ['valence', 'arousal', 'full', 'emotion']
    for time in timing:
        for cat in category:
            best_transform = best_data[time][cat]['db']
            best_level = best_data[time][cat]['level']
            if time == 'early':
                early.append([best_transform, best_level])
            else:
                late.append([best_transform, best_level])
    return early, late
        