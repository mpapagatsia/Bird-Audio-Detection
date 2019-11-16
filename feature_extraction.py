import pandas as pd
import numpy as np
import os.path
import librosa
import scipy.io.wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import librosa.display
import sklearn
import _pickle
from sklearn.preprocessing import QuantileTransformer
import sys
from scipy.signal import butter

#fix seed for reproducibility
np.random.seed(7)

labels_csv = ["warblrb10k.csv", "ff1010bird.csv", "BirdVox-DCASE-20k.csv"]

max_frames = 431

#butterworth highpass filter :
#The filter reduces low frequency energy improving
#signal-to-noise ratio for frequency bands relevant to bird sounds.
#(1st technical report DCASE 2018)
def butter_highpass(cutoff, fs, order=2):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
            return b, a


#extract librosa mfcc features
def mfcc(item,directory):
    #if an item's id is given as input
    if isinstance(item, str) :
        audio = str(item)
    else:
        audio = str(item.itemid)

    #print("Item: ", audio)
    
    try:
        signal, sampling_rate = librosa.load(os.path.join(directory+audio+'.wav'), sr=44100)

        #apply a butterworth highpass filter to improve SNR for bands relevant to bird sounds
        if item.hasbird == 1:
            b, a = butter_highpass(2000, sampling_rate)
            signal = scipy.signal.filtfilt(b, a, signal)

        """f = plt.figure(1)
        plt.plot(signal)
        f.show()
        input()"""

        #resample the signal to reduce amount of data
        signal = librosa.resample(signal, sampling_rate, 22050)

        sampling_rate = 22050

        duration = librosa.get_duration(y=signal)

        #adjust hop length and n_fft in order to have a fixed size vestor
        #fixed is (431,39) prodused by the majority of th file having 10s length 
        if duration != 10:
            #print(directory)
            hop = int(duration / 0.019)
            fft = hop * 4 #75% overlap
            mfcc = librosa.feature.mfcc(y=signal, n_mfcc=13, hop_length = hop, n_fft = fft)
        else:
            mfcc = librosa.feature.mfcc(y=signal, sr=sampling_rate, n_mfcc=13)
        
        delta = librosa.feature.delta(mfcc)
        delta_delta = librosa.feature.delta(mfcc, order=2)
        
        #plot the features
        """fig2 = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, y_axis='mel', x_axis='time', sr=sampling_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()"""

        features = np.concatenate((mfcc,delta,delta_delta), axis=0)

        #plt.hist(features[0])
        #plt.show()
        #input()

        features = features.transpose()

        #Feature standardization zero mean unit variance
        scaler = sklearn.preprocessing.StandardScaler()
        features = scaler.fit_transform(features)

        #Transform data to have uniform distribution
        #proposed by EUSPIPCO : Bird Activity Detection Using Probabilistic Sequence Kernels
        qt = QuantileTransformer(n_quantiles=200, random_state=0, output_distribution='normal')
        features = qt.fit_transform(features)

        
        #pad rows with the mean values to fit the (431,39) size
        if features.shape[0] != max_frames :
            means = np.mean(features, axis=0)
            means = means.reshape(1,-1) 
            means = np.tile(means, (max_frames - features.shape[0], 1))
            features = np.concatenate((features,means), axis=0)
    
        """features = features.transpose()
        e = plt.figure(2)
        plt.hist(features[0],bins=39,alpha=0.5,ec='black')
        e.show()
        input()
        """

        """fig3 = plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc, y_axis='mel', x_axis='time', sr=sampling_rate)
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC normilized')
        plt.tight_layout()
        fig3.show()"""

    except Exception as e:
        print("Error while processing the file", e)

    #input()
    return features

import subprocess
import HTK
from HTK import HTKFile
import soundfile as sf

def opensmile_mfcc(item,directory):
    opensmile_path = "/home/mpapagatsia/opensmile-2.3.0/"

    if isinstance(item, str) :
        audio = str(item)
    else:
        audio = str(item.itemid)

    command = opensmile_path+"./SMILExtract -C "+opensmile_path+"config/MFCC12_E_D_A_Z.conf -I "+directory+audio+".wav"+" -O /home/mpapagatsia/output.mfcc.htk"

    output= subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)

    htk_reader = HTKFile()
    htk_reader.load("/home/mpapagatsia/output.mfcc.htk")
    features = np.array(htk_reader.data)

    print(features.shape)
    
    return features

def chroma_cens_lib(item,directory, extention = '.wav'):
    if isinstance(item, str) :
        audio = str(item)
        bird = 0
    else:
        audio = str(item.itemid)
        bird = item.hasbird
    
    #print(audio)

    try:
        if extention == '.wav':
            signal, sampling_rate = librosa.load(os.path.join(directory+audio+extention), sr=44100)
        else:
            signal, sampling_rate = sf.read(directory+audio)
    except Exception as e:
        print("Error while processing the file", e)
    
    if bird == 1:
        b, a = butter_highpass(2000, sampling_rate)
        signal = scipy.signal.filtfilt(b, a, signal)
        
    signal = librosa.resample(signal, sampling_rate, 22050)
    sampling_rate = 22050

    duration = librosa.get_duration(y=signal)
    
    #adjust hop length in order to have a fixed size vestor
    if duration != 10:
        if duration > 10:
            chromagram = librosa.feature.chroma_cens(y=signal, sr=sampling_rate,hop_length=1024)
        else:
            chromagram = librosa.feature.chroma_cens(y=signal, sr=sampling_rate,hop_length=512)
    else:
        chromagram = librosa.feature.chroma_cens(y=signal, sr=sampling_rate,hop_length=512)
        
    
    features = chromagram.transpose()
    

    return features

def chroma_cqt_lib(item,directory, extention = '.wav'):
    if isinstance(item, str) :
        audio = str(item)
        bird = 0
    else:
        audio = str(item.itemid)
        bird = item.hasbird
    
    #print(audio)

    try:
        if extention == '.wav':
            signal, sampling_rate = librosa.load(os.path.join(directory+audio+extention), sr=44100)
        else:
            signal, sampling_rate = sf.read(directory+audio)
    except Exception as e:
        print("Error while processing the file", e)
    
    if bird == 1:
        b, a = butter_highpass(2000, sampling_rate)
        signal = scipy.signal.filtfilt(b, a, signal)
        
    signal = librosa.resample(signal, sampling_rate, 22050)
    sampling_rate = 22050

    duration = librosa.get_duration(y=signal)
    
    #adjust hop length to have a fixed size vestor
    if duration != 10:
        if duration > 10:
            chromagram = librosa.feature.chroma_cqt(y=signal, sr=sampling_rate,hop_length=1024)
        else:
            chromagram = librosa.feature.chroma_cqt(y=signal, sr=sampling_rate,hop_length=512)
    else:
        chromagram = librosa.feature.chroma_cqt(y=signal, sr=sampling_rate,hop_length=512)
    
    
    features = chromagram.transpose()
    
    #Normal Distribution
    qt = QuantileTransformer(n_quantiles=200, random_state=0, output_distribution='normal')
    features = qt.fit_transform(features)
    #print(features.shape)
    return features

#(not used in this study)
def opensmile_prosodic(item, directory):
    opensmile_path = "/home/mpapagatsia/opensmile-2.3.0/"

    if isinstance(item, str) :
        audio = str(item)
    else:
        audio = str(item.itemid)
    
    command = opensmile_path+"./SMILExtract -C "+opensmile_path+"config/prosodyShs.conf -I "+directory+audio+".wav"+" -csvoutput /home/mpapagatsia/prosody.csv"
    
    output= subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    
    features = pd.read_csv("/home/mpapagatsia/prosody.csv", delimiter=';')
    #print(features.head())
    #features = np.mean(features, axis=0)
    print(features.shape)

    
    return features

def opensmile_chroma(item, directory):
    opensmile_path = "/home/mpapagatsia/opensmile-2.3.0/"

    if isinstance(item, str) :
        audio = str(item)
    else:
        audio = str(item.itemid)
    
    command = opensmile_path+"./SMILExtract -C "+opensmile_path+"config/chroma_fft.conf -I "+directory+audio+".wav"+" -O /home/mpapagatsia/chroma.csv"
    
    output= subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT)
    
    #change path!!!
    features = pd.read_csv("/home/mpapagatsia/chroma.csv", delimiter=';')
    
    features = features.values 
    
    qt = QuantileTransformer(n_quantiles=200, random_state=0, output_distribution='normal')
    features = qt.fit_transform(features)

    return features

#split in train and test dataset
#use 15% of each dataset: of which 20% for testing
#arg : path the path where the labels are stored
def dataset_split(path):
    path = path + "/labels/"
    #configure labels dir as input argument

    for k in range(len(labels_csv)):
        df = pd.read_csv(path+labels_csv[k])

        df_groups = [x for _, x in df.groupby(df['hasbird'])]

        print(labels_csv[k])
        print("Statistics in total: No bird: %d , Has bird: %d " %(df_groups[0].shape[0], df_groups[1].shape[0]))
        
        print("Use 15 per cent of each dataset: "   )
        df_groups[0] = df_groups[0].sample(frac=0.15) #0.15 #0.08 for hmm
        df_groups[1] = df_groups[1].sample(frac=0.15) #0.15 #0.08 for hmm
        
        test_no = df_groups[0].sample(frac = 0.3) #0.3 #0.3

        test_has =  df_groups[1].sample(frac = 0.3) #0.3 #0.3

        test_df = pd.concat([test_no, test_has], axis = 0)
        print("Test shape: ", test_df.shape)
        
        test_df.reset_index(drop=True, inplace=True)
        
        #suffle and drop old index
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        
        #Write TEST files
        test_df.to_csv(path+'/test/'+labels_csv[k], encoding='utf-8')
        
        #Train samples
        train_no = pd.concat([df_groups[0], test_no]).drop_duplicates(keep=False)
        
        train_has = pd.concat([df_groups[1], test_has]).drop_duplicates(keep=False)
        
        #print("Train has shape: ", train_has.shape)
        train_df = pd.concat([train_no, train_has], axis=0)
        print("Train shape: ", train_df.shape)
        
        train_df.reset_index(drop=True, inplace=True)
        #suffle and drop old index
        train_df = train_df.sample(frac=1).reset_index(drop=True)
        
        #Write TRAIN files
        train_df.to_csv(path+'/train/'+labels_csv[k], encoding='utf-8')
        print("\n")


def feature_extraction(dir_path, datasets = labels_csv, f_type=''):
    #configure data path
    labels = dir_path + "/labels/"
    split_path = [labels+ "train/", labels + "test/"]
    to_store = dir_path + "/features/" + f_type + '/'

    for path in split_path:
        if not os.path.isdir(to_store + path.replace(labels, '')):
            os.makedirs(to_store + path.replace(labels, ''))

    if f_type == '':
        print("No features type specified!")
        sys.exit()
    elif f_type == 'opensmile_mfcc':
        f = opensmile_mfcc
    elif f_type == 'mfcc':
        f = mfcc
    elif f_type == 'cens':
        f = chroma_cens_lib
    elif f_type == 'cqt':
        f = chroma_cqt_lib
    elif f_type == 'opensmile_chroma':
        f = opensmile_chroma
    else:
        print("Wrong features type!")
        sys.exit()

    for path in split_path:
        for k in range(len(datasets)):
            df = pd.read_csv(path+datasets[k])
            
            ids = pd.DataFrame()
            
            ids[['itemid','hasbird']] = df[['itemid','hasbird']]
            
            df_features = df.apply(eval('f') , axis=1, directory = dir_path+"/audio/"+datasets[k].replace('.csv','/'))
            
            df_flatten = pd.DataFrame()

            for j in range(len(df_features)):
                temp2 = df_features[j].flatten()
                
                vector = pd.DataFrame([temp2])

                df_flatten= df_flatten.append(vector,ignore_index=True)
            
            ids.reset_index(drop=True, inplace=True)
            df_flatten.reset_index(drop=True, inplace=True)
            
            ids = pd.concat([ids,df_flatten], axis=1)               
            
            ids.to_csv(to_store + path.replace(labels, '')+datasets[k], encoding='utf-8')

#feature_extraction("/home/mpapagatsia/Documents/data_birds/", f_type='cqt')


#extract features from different dataset. (not used in the study)
import os
def extract_test_features():
    path = "/home/mpapagatsia/Documents/data_birds/multilabel-bird-species-classification-nips2013/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/"
    #files = os.listdir(path)
    label_p = "/home/mpapagatsia/Documents/data_birds/multilabel-bird-species-classification-nips2013/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/" 
    
    #clean dataset ---
    df_flatten = pd.DataFrame()
    
    df = pd.read_excel(label_p+'nips4b_birdchallenge_train_labels.xls')
    
    df.columns = df.iloc[1]
    df = (df.drop([0,1])).copy()
    df = df.drop(df.index[len(df)-1])
    df.loc[df['Empty'] == 1, 'Empty'] = 0
    #end cleaning ---

    #where to store features
    ids = pd.DataFrame()
    ids[['itemid','hasbird']] = df[['Filename','Empty']]
    
    #specific
    ids = ids.fillna(1)

    for item in ids['itemid']:
        if item.endswith('.wav'):
            item = item.replace('.wav', '')
            f = chroma_cens_lib(item,path,extention='.wav')
            temp = f.flatten()
            vector = pd.DataFrame([temp])
            df_flatten= df_flatten.append(vector,ignore_index=True)
    
    ids.reset_index(drop=True, inplace=True)
    df_flatten.reset_index(drop=True, inplace=True)

    ids = pd.concat([ids,df_flatten]  , axis=1)
    
    to_store = "/home/mpapagatsia/Documents/data_birds/features/testing/test/"
    ids.to_csv(to_store +'test.csv', encoding='utf-8')
    
#extract_test_features()
