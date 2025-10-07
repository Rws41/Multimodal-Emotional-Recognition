import os
import pywt
import re
import helper
import explorer as ex
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy.stats as st
from pathlib import Path
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter

##Classes for managing x and y data values
#Manage data retrieval, data filtering, feature extraction, and data fusion
class manager():
    def __init__(self):
        root = Path.cwd()
        self.ecg_path = root / 'data' / 'ECG'
        self.gsr_path = root / 'data' / 'GSR'
        self.ecg_raw_image_path = root / 'images' / 'raw' / 'ECG'
        self.gsr_raw_image_path = root / 'images' / 'raw' / 'GSR'
        self.ecg_filtered_image_path = root / 'images' / 'filtered' / 'ECG'
        self.gsr_filtered_image_path = root / 'images' / 'filtered' / 'GSR'
        self.ecg = np.zeros((252, 5000))
        self.ecg_ids = np.zeros((252, 3))
        self.gsr = np.zeros((252, 5000))
        self.gsr_ids = np.zeros((252, 3))

    #Function to load data from a .mat file
    def data_load(self, file, ecg = True):
        mat_data = loadmat(file)
        if ecg:
            data = mat_data['ECGdata']

        else:
            data = mat_data['GSRdata']
        data_array = np.array(data).flatten()
        return data_array

    #Getting the data. The first three columns are tags providing info. Session, Participant, Video
    #If data exists just load the file.
    def raw_data_compile(self, path, bio, ecg = True):
        new_bio = np.zeros((bio.shape[0], bio.shape[1]+3))
        i=0
        for file in path.iterdir():
            ids = self.tag_data(file)
            if ecg:
                bio_data = self.data_load(file, True)
                self.data_plotter(bio_data, ids, self.ecg_raw_image_path,)
            else:
                bio_data = self.data_load(file, False)
                self.data_plotter(bio_data, ids, self.gsr_raw_image_path, ecg)

            bio_data = np.concatenate((ids, bio_data))
            new_bio[i, :] = bio_data
            i += 1
        return new_bio

    #Add check for data existence and return that if it exists
    def raw_ecg_data_get(self):
        if os.path.exists('./data/ecg_data_raw.csv'):
            return np.genfromtxt('./data/ecg_data_raw.csv', delimiter=',')
        else:
            self.ecg_raw = self.raw_data_compile(self.ecg_path, self.ecg, True)
            return self.ecg_raw
        
    def raw_gsr_data_get(self):
        if os.path.exists('./data/gsr_data_raw.csv'):
            return np.genfromtxt('./data/gsr_data_raw.csv', delimiter=',')
        else:
            self.gsr_raw = self.raw_data_compile(self.gsr_path, self.gsr, False)
            return self.gsr_raw
    
    #Filters
    #Order of filter will be set at 4. Seems typical is 2-5. Too high and get distortions, too low and get noise
    def butterworth_band_filter(self, data, low_thresh, high_thresh, fs):
        nyquist = fs * 0.5
        low = low_thresh / nyquist
        high = high_thresh/nyquist
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data)
        return filtered

    def notch_filter(self, data):
        b, a = iirnotch(50, 30, 256)
        filtered = filtfilt(b, a, data)
        return filtered



    #The ECG data is pre-filtered with a second-order Chebyshev low pass filter (LPF) with a corner frequency slightly smaller than the Nyquist frequency
    #and a second-order Chebyshev high pass filter (HPF) with a corner frequency of 0.5 Hz to minimize the effect of environmental noise and muscle movement before data storage. 
    #The interference noise is minimized during signal acquisition;
    def ecg_filter(self, data):
        if os.path.exists('./data/ecg_filtered.csv'):
            return np.genfromtxt('./data/ecg_filtered.csv', delimiter=',')
        else:
            filtered_data = np.zeros(data.shape)
            for i in range(data.shape[0]):
                ids = data[i,:3]
                filtered_signal = self.butterworth_band_filter(data = data[i, 3:], low_thresh=0.03, high_thresh=60, fs=256)
                filtered_signal = self.notch_filter(filtered_signal)
                filtered_data[i, :3] = ids
                filtered_data[i, 3:] = filtered_signal
                #skip filtering due to prefiltered?
                self.data_plotter(filtered_signal, ids, self.ecg_filtered_image_path, ecg=True, raw=False)
            return filtered_data


    def savgol_filter(self, data):
            return savgol_filter(data, window_length=15, polyorder=3)
    
    def gsr_filter(self, data):
        if os.path.exists('./data/gsr_filtered.csv'):
            return np.genfromtxt('./data/gsr_filtered.csv', delimiter=',')
        else:
            for i in range(data.shape[0]):
                ids = data[i, :3]
                trial_data = data[i, 3:]
                filtered_trial = self.savgol_filter(trial_data)
                data[i, 3:] = filtered_trial
                #Plot the data
                self.data_plotter(filtered_trial, ids, self.gsr_filtered_image_path, ecg=False, raw=False)
            return data
    
    def load_and_filter(self):
        ecg = self.raw_ecg_data_get()
        gsr = self.raw_gsr_data_get()
        np.savetxt('./data/ecg_data_raw.csv', ecg, delimiter=',')
        np.savetxt('./data/gsr_data_raw.csv', gsr, delimiter=',')

        ###Filtering the data
        ecg_filtered = self.ecg_filter(ecg)
        gsr_filtered = self.gsr_filter(gsr)
        np.savetxt('./data/ecg_filtered.csv', ecg_filtered, delimiter=',')
        np.savetxt('./data/gsr_filtered.csv', gsr_filtered, delimiter=',')
        return ecg_filtered, gsr_filtered
    
    #Getting tag from file name to connect to participants, videos,and sessions
    def tag_data(self, file):
        filename = file.name
        reg = re.compile(r'\d{1,2}')
        match = reg.findall(filename)
        match = [int(element) for element in match]
        tag_data = np.array(match)
        return tag_data

    def data_plotter(self, data, ids, path, ecg=True, raw=True):
        ids = [int(x) for x in ids]
        if raw:
            if ecg:
                title = f"P {ids[1]} S {ids[0]} V{ids[2]} ECG Raw"
            else:
                title = f"P {ids[1]} S {ids[0]} V{ids[2]} GSR Raw"
        else:
            if ecg:
                title = f"P {ids[1]} S {ids[0]} V{ids[2]} ECG Filtered"
            else:
                title = f"P {ids[1]} S {ids[0]} V{ids[2]} GSR Filtered"

        plt.plot(data)
        plt.title(title)
        plt.savefig(str(path) + '/' + title + '.png')
        plt.close()
        return
    
    ###Feature extractor class?

    #Heart Rate Variability evaluation
    def HRV(self, data, ids, plot=False):
        _, r_peaks = nk.ecg_peaks(data, sampling_rate=256, method='emrich2023')
        peaks = r_peaks['ECG_R_Peaks']

        #Breaking data into RR segments and getting HRV metrics
        rr_intervals = np.diff(peaks) / 256
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        rr_diff = np.diff(rr_intervals)
        rmssd = np.sqrt(np.mean(rr_diff ** 2))

        num_intervals = len(rr_intervals)
        rr_diff = np.abs(np.diff(rr_intervals))
        nn50 = np.sum(rr_diff > 0.05)
        pnn50 = nn50 / num_intervals

        if plot:
            quality = nk.ecg_quality(data, sampling_rate=256, method='zhao2018', approach='fuzzy')
            plt.figure()
            plt.plot(data)
            plt.scatter(peaks, data[peaks], color='red')
            plt.title("R Peak Identification")
            plt.savefig(f'./images/filtered/ECG/Peaks/Participant {ids[1]} Session {ids[0]} Video {ids[2]}.png')
            plt.close()
            return quality, [rr_mean, rr_std, rmssd, pnn50]

        return [rr_mean, rr_std, rmssd, pnn50]

    def flatten(self, x):
        for i in x:
            if isinstance(i, list):
                yield from self.flatten(i)
            else:
                yield i

    #Frequency analysis of high and low frequency bands
    def frequency_band_analysis(self, low_frequency, high_frequency):
        low_freq_skew = st.skew(low_frequency)
        high_freq_skew = st.skew(high_frequency)
        lf_range = np.max(low_frequency) - np.min(low_frequency)
        hf_range = np.max(high_frequency) - np.min(high_frequency)
        return [low_freq_skew, high_freq_skew, lf_range, hf_range]

    def frequency_overall_analysis(self, frequencies):
        overall_kurtosis = st.kurtosis(frequencies)
        overall_skew = st.skew(frequencies)
        overall_standard_dev = np.std(frequencies)
        overall_mean = np.mean(frequencies)
        return [overall_kurtosis, overall_skew, overall_standard_dev, overall_mean]

    #Fast Fourier Transform Analysis
    def FFT_analysis(self, data, ids, plot=False):
        fft = np.fft.fft(data)
        fft_freqs = np.fft.fftfreq(len(data), 1 / 256)
        pos_freqs = fft_freqs[fft_freqs >= 0]
        mag = np.abs(fft)
        mag_sq = np.abs(fft) ** 2

        #Power Spectral Density Analysis
        psd = mag_sq / (len(data) * 256)
        pos_psd = psd[fft_freqs >= 0]

        #Spectral Centroid
        spectral_centroid = np.sum(pos_freqs * pos_psd) / np.sum(pos_psd)

        #Activity level in ECG bands of signficance
        lf_mask = (pos_freqs >= 0.04) & (pos_freqs < 0.15)
        hf_mask = (pos_freqs >= 0.15) & (pos_freqs < 0.4)

        #Ratio of low to high power frequency power
        power_ratio = np.sum(pos_psd[lf_mask]) / np.sum(pos_psd[hf_mask])

        #Isolating low and high frequency bands in question and analysis
        low_frequency_band = pos_psd[lf_mask]
        high_frequency_band = pos_psd[hf_mask]
        band_stats = self.frequency_band_analysis(low_frequency_band, high_frequency_band)

        #Analysis of overall frequency
        overall_frequency_stats = self.frequency_overall_analysis(pos_psd)

        if plot == True:
            plt.plot(pos_freqs, pos_psd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.xlim([0, 0.4])
            plt.fill_between(pos_freqs, pos_psd, where=(pos_freqs >= 0.04) & (pos_freqs < 0.15),
                             color='lightpink', alpha=0.5, label='LF')
            plt.fill_between(pos_freqs, pos_psd, where=(pos_freqs >= 0.15) & (pos_freqs <= 0.4),
                             color='lightgreen', alpha=0.5, label='HF')
            plt.title("ECG Power Spectral Density - Low and High Frequency Bands")
            plt.savefig(f'./images/filtered/ECG/FFT/Participant {ids[1]} Session {ids[0]} Video {ids[2]} PSD.png')
            plt.close()

            plt.plot(pos_freqs, pos_psd)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Power Spectral Density")
            plt.title("ECG Power Spectral Density Full Range")
            plt.savefig(f'./images/filtered/ECG/FFT/Participant {ids[1]} Session {ids[0]} Video {ids[2]} PSD_Full.png')
            plt.close()

            plt.plot(fft_freqs, mag)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude")
            plt.title("ECG Frequency Spectrum")
            plt.savefig(f'./images/filtered/ECG/FFT/Participant {ids[1]} Session {ids[0]} Video {ids[2]} Frequency Spectrum.png')
            plt.close()

        return [spectral_centroid, power_ratio, band_stats, overall_frequency_stats]

    #Retrieving Descriptive Statistics
    def stat_getter(self, data):
        avg = np.mean(data)
        stn_dev = np.std(data)
        kur = st.kurtosis(data)
        skew = st.skew(data)
        variance = np.var(data)
        return [avg, stn_dev, kur, skew, variance]
    
    
    def ecg_feature_extraction(self, data, db, level, labels, plot=False, FFT=False):
        if os.path.exists('./data/ecg_features.csv'):
            return pd.read_csv('./data/ecg_features.csv')
        else:
            if plot:
                quality_ratings = []
            if FFT:
                ecg_features = np.zeros((data.shape[0], 22))
            else:
                ecg_features = np.zeros((data.shape[0], len(labels)))
            for i in range(data.shape[0]):
                ids = data[i,:3]
                ecg_features[i, :3] = ids
                #baseline = data[i, 3:1280]
                emotion = data[i, 1283:]
                #Statistical Breakdown of data
                ecg_stats = self.stat_getter(emotion)

                #Heart Rate Varaibility Information
                if plot:
                    quality, HRV_metrics = self.HRV(emotion, ids, plot=plot)
                    quality_ratings.append(quality)
                else:
                    HRV_metrics = self.HRV(emotion, ids, plot=plot)
                #Frequency Analysis
                if FFT:
                    features = []
                    features.append(HRV_metrics)
                    FFT_metrics = self.FFT_analysis(emotion, ids, plot=plot)
                    #Condensing features into a dataframe
                    features.append(FFT_metrics)
                    features.append(ecg_stats)
                    features_full = list(self.flatten(features))
                    ecg_features[i, 3:] = features_full
                    ecg_features_df = pd.DataFrame(data=ecg_features, columns=labels)
                else:
                    # Discrete Wavelet Transform
                    features = self.dwt_analysis(HRV_metrics, ecg_stats, emotion, db, level)
                    #Condensing features into a dataframe
                    ecg_features[i, 3:] = features
                    ecg_features_df = pd.DataFrame(data=ecg_features, columns=labels)
            return ecg_features_df
        
    def dwt_analysis(self, hrv, ecg_stats, emotion, db, level):
        coeff = pywt.wavedec(emotion, wavelet=db, level=level)
        energies = [np.sum(np.square(c)) for c in coeff]
        means = [np.mean(c) for c in coeff]
        stdev = [np.std(c) for c in coeff]
        max_in_coeff = [np.max(c) for c in coeff]
        min_in_coeff = [np.min(c) for c in coeff]
        entropy = -np.sum((e / np.sum(energies)) * np.log(e / np.sum(energies)) for e in energies)
        features = np.concatenate((hrv, energies, means, stdev, max_in_coeff, min_in_coeff, [entropy], ecg_stats))
        return features

    def gsr_feature_extraction(self, data, labels, db, level):
        if os.path.exists('./data/gsr_features.csv'):
            return pd.read_csv('./data/gsr_features.csv')
        else:
            gsr_features = np.zeros((data.shape[0], len(labels)))
            for i in range(data.shape[0]):
                ids = data[i,:3]
                gsr_features[i,:3] = ids
                emotion = data[i, 1283:]

                #Discrete Wavelet Transform
                coeff = pywt.wavedec(emotion, wavelet=db, level=level)

                energies = [np.sum(np.square(c)) for c in coeff]
                means = [np.mean(c) for c in coeff]
                stdev = [np.std(c) for c in coeff]
                max_in_coeff = [np.max(c) for c in coeff]
                min_in_coeff = [np.min(c) for c in coeff]
                entropy = -np.sum((e / np.sum(energies)) * np.log(e / np.sum(energies)) for e in energies)

                features = self.stat_getter(emotion)
                gsr_range = np.max(emotion) - np.min(emotion)
                features = np.concatenate((features, [gsr_range], energies, means, stdev, max_in_coeff, min_in_coeff, [entropy]))
                gsr_features[i, 3:] = features

            gsr_features_df = pd.DataFrame(data=gsr_features, columns = labels)
            return gsr_features_df
    
    def analysis_feature_extraction(self, filtered, results, gsr=True):
        if gsr:
            features = []
            for x in results:
                db = x[0]
                level = x[1]
                features.append(self.gsr_feature_extraction(filtered, labels=helper.label_generator(level, gsr), db=db, level=level))
            return features
        else:
            features = []
            for x in results:
                db = x[0]
                level = x[1]
                if db == 'FFT':
                    features.append(self.ecg_feature_extraction(filtered, labels=helper.fft_label_generator(), db=db, level=level, FFT=True))
                else:
                    features.append(self.ecg_feature_extraction(filtered, labels = helper.label_generator(level, gsr), db=db, level=level))
            return features
        
    def data_fusion(self, ecg, gsr, y):
        drop_cols = ['Session ID', 'Participant Id', 'Video ID']

        ecg_response = pd.merge(ecg, y, how='inner',
                                         left_on=['Session', 'Participant', 'Video'],
                                         right_on=['Session ID', 'Participant Id', 'Video ID'])

        ecg_response.drop(ecg_response[drop_cols], axis=1, inplace=True)

        gsr_response = pd.merge(gsr, y, how='inner',
                                         left_on=['Session', 'Participant', 'Video'],
                                         right_on=['Session ID', 'Participant Id', 'Video ID'])
        gsr_response.drop(gsr_response[drop_cols], axis=1, inplace=True)



        all_features = pd.merge(ecg, gsr, how='inner', on=['Session', 'Participant', 'Video'])
        all_features_response = pd.merge(all_features, y, how='inner',
                                         left_on=['Session', 'Participant', 'Video'],
                                         right_on=['Session ID', 'Participant Id', 'Video ID'])
        drop_cols = ['Session ID', 'Participant Id', 'Video ID']
        all_features_response.drop(all_features_response[drop_cols], axis=1, inplace=True)
        return ecg_response, gsr_response, all_features_response
    
    def fusion_scaling_splitting(self, gsr_features, ecg_features, responses, is_emotion, labels, selected, unselected, early=True):
        if early:
            results = []
            for ecg, gsr, response, emotion, label in zip(ecg_features, gsr_features, responses, is_emotion, labels):
                _,_, y = self.data_fusion(ecg, gsr, response)
                results.append(helper.train_test_scale_split(y, selected, unselected, emotion, label))
            return results
        else:
            ecg_results = []
            gsr_results = []
            for ecg, gsr, response, emotion, label in zip(ecg_features, gsr_features, responses, is_emotion, labels):
                j, k, _ = self.data_fusion(ecg, gsr, response)
                ecg_results.append(helper.train_test_scale_split(j, selected, unselected, emotion, label))
                gsr_results.append(helper.train_test_scale_split(k, selected, unselected, emotion, label))
            return ecg_results, gsr_results




#Tool to help manage the y values of the data set and categorizing the self response data into classification categories
class y_response_handler():
    def __init__(self):
        pass

    def response_loading(self):
        responses = pd.read_csv('./data/Self-annotation Multimodal.csv')
        return responses


    def emotion_coding(self, df):
        df_copy = df.copy()
        emotion_cols = ['Happy', 'Sad', 'Fear', 'Anger', 'Neutral', 'Disgust', 'Surprised']
        for emotion in emotion_cols:
            df_copy[emotion] = df_copy[emotion].apply(self.emotion_coding_helper)
        slimmed_df = df_copy.copy()
        slimmed_df.drop(slimmed_df.columns[3:9], axis=1, inplace=True)  #Drop columns 3 to 9
        slimmed_df.drop(slimmed_df.columns[-1], axis=1, inplace=True)
        emotions = slimmed_df.iloc[:,3:]
        max_cols = emotions.idxmax(axis=1)
        trial_responses = pd.concat([df.iloc[:,0:3], max_cols], axis=1)
        return trial_responses

    def emotion_coding_helper(self, emotion):
        if emotion == 'VeryLow':
            return 0
        if emotion == 'Low':
            return 1
        if emotion == 'Moderate':
            return 2
        if emotion == 'High':
            return 3
        if emotion == 'VeryHigh':
            return 4

    def val_arousal_helper(self, val):
        if val < 5:
            return 0
        if val >= 5:
            return 1

    def valence_classifier(self, x, y):
        if x==1:
            return 'HV'
        if x==0:
            return 'LV'

    def arousal_classifier(self, x, y):
        if y==1:
            return 'HA'
        if y==0:
            return 'LA'

    def val_arousal_classifier(self, x, y):
        if x==1 and y==1:
            return 'HVHA'
        if x==1 and y==0:
            return 'HVLA'
        if x==0 and y==1:
            return 'LVHA'
        if x==0 and y==0:
            return 'LVLA'

    def val_arousal_coding(self, df):
        cols = ['Valence level', 'Arousal level']
        for col in cols:
            df[col] = df[col].apply(self.val_arousal_helper)
        slimmed_df = df.copy()
        slimmed_df.drop(slimmed_df.columns[3:6], axis=1, inplace=True)
        slimmed_df.drop(slimmed_df.columns[5:], axis=1, inplace=True)

        subset = slimmed_df.iloc[:, 3:5]
        slimmed_df['Valence_Classification'] = subset.apply(lambda row: self.valence_classifier(row['Valence level'], row['Arousal level']), axis=1)
        slimmed_df['Arousal_Classification'] = subset.apply(lambda row: self.arousal_classifier(row['Valence level'], row['Arousal level']), axis=1)
        slimmed_df['Val_Arousal_Classification'] = subset.apply(lambda row: self.val_arousal_classifier(row['Valence level'], row['Arousal level']), axis=1)
        slimmed_df.drop(columns=['Valence level', 'Arousal level'], inplace=True)
        return slimmed_df
    
    

    def response_coding(self, exploration=False):
        responses = self.response_loading()
        #EDA
        if exploration==True:
            ex.exploratory_analysis(responses)

        ###Coding data to emotion and valence/arousal categories
        emotion_responses = self.emotion_coding(responses)
        emotion_responses.to_csv('./data/emotion_responses.csv', index=False)

        val_arousal_responses = self.val_arousal_coding(responses)
        val_arousal_responses.to_csv('./data/val_arousal_responses.csv', index=False)

        return emotion_responses, val_arousal_responses



    










