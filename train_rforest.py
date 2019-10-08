from sklearn import svm
import pandas as pd
import numpy as np
import os.path
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import random 
from sklearn import metrics
import _pickle
import matplotlib.pyplot as plt


labels_csv = ["warblrb10k.csv", "ff1010bird.csv", "BirdVox-DCASE-20k.csv"]
size = 40

#max_frames = 431 #431 librosa , 993 chroma open, 998 mfcc open

random.seed(7)

def testing(test_folds, model, path, components, max_frames):
    max_size = components * max_frames
    test_paths = [path+"/test/"]
    features = np.asarray(())
    labels = np.asarray(())

    for p in range(len(test_paths)):
        for folds in test_folds:
            for df_test in pd.read_csv(test_paths[p] +folds,chunksize=size):
                df = df_test.drop(['Unnamed: 0', 'itemid'], axis=1)
                
                if labels.size == 0:
                    labels = df['hasbird'].values
                else:
                    labels = np.hstack((labels,df['hasbird'].values))
                
                df = df.drop(['hasbird'], axis=1)

                df = df.values
                
                for i in range(len(df)):
                    temp = df[i][np.isfinite(df[i])]
                    
                    if len(temp) == max_size:
                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
                    elif len(temp) < max_size:
                        temp = temp.reshape(-1,components)
                        
                        means = np.mean(temp, axis=0)
                        means = means.reshape(1,-1)
                        means = np.tile(means, (max_frames - temp.shape[0], 1))
                        temp = np.concatenate((temp,means), axis=0)
                        
                        temp = temp.reshape(1,-1)

                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
                    else:
                        temp = temp.reshape(-1,components)
                        temp = np.delete(temp, np.s_[max_frames::], 0)
                        temp = temp.reshape(1,-1)
                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
    
    print("TEST...")
    
    print(features.shape)
    print(labels.shape)
    
    pred = model.predict(features)
    rf_probs = model.predict_proba(features)[:, 1]
    
    #acc = accuracy_score(labels, pred)
    #print("Accuracy on tests: ", acc)

    fpr, tpr, thresholds = metrics.roc_curve(labels, rf_probs)
    
    auc = metrics.auc(fpr, tpr)
    print("AUC (labels) test: ", metrics.roc_auc_score(labels, pred))
    print("AUC (probs) test: ", auc)
    #print("True positive rate: ", tpr, "False positive rate: ", fpr)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    #plt.show()
    return auc
    

from sklearn.ensemble import RandomForestClassifier

def training(train_folds, test_fold, path, components, max_frames):
    max_size = components * max_frames
    paths = [path+"/train/"]

    features = np.asarray(())
    labels = np.asarray(())

    for p in range(len(paths)):
        for folds in train_folds:
            for df_train in pd.read_csv(paths[p] +folds,chunksize=size):
                df = df_train.drop(['Unnamed: 0', 'itemid'], axis=1)

                if labels.size == 0:
                    labels = df['hasbird'].values
                else:
                    labels = np.hstack((labels,df['hasbird'].values))
                
                df = df.drop(['hasbird'], axis=1)

                df = df.values
                
                for i in range(len(df)):
                    temp = df[i][np.isfinite(df[i])]

                    if len(temp) == max_size:
                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
                    elif len(temp) < max_size:
                        temp = temp.reshape(-1,components)
                        
                        means = np.mean(temp, axis=0)
                        means = means.reshape(1,-1)
                        means = np.tile(means, (max_frames - temp.shape[0], 1))
                        temp = np.concatenate((temp,means), axis=0)
                        
                        temp = temp.reshape(1,-1)
                        

                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
                    else:
                        temp = temp.reshape(-1,components)
                        temp = np.delete(temp, np.s_[max_frames::], 0)
                        temp = temp.reshape(1,-1)
                        if features.size == 0:
                            features = temp
                        else:
                            features = np.vstack((features,temp))
    
    print("TRAIN...")
    
    model = RandomForestClassifier(n_estimators=300, 
                                    bootstrap = True,
                                    max_features = 'sqrt').fit(features, labels.ravel())

    ftype = path.split("/")
    ftype = [i for i in ftype if i]
    ftype = ftype[-1]                              
    _pickle.dump(model, open(ftype+''.join([i[0] for i in train_folds])+'model.rf','wb'))
   

import statistics 
def cross_val(data_path, feat_type,comp_num, frames):
    
    kf = KFold(n_splits=3)
    auc = []

    print("Training with Random Forest...")
    for train_index, test_index in kf.split(labels_csv):
        csv = np.array(labels_csv)
        print("Train with: ",csv[train_index]," Test with: ", csv[test_index])
        
        train = csv[train_index]
        test = csv[test_index]
        
        training(train, ["Empty"], path=data_path + '/features/'+feat_type+'/', components=comp_num, max_frames=frames)
        model = _pickle.load(open(feat_type+''.join([i[0] for i in train])+'model.rf','rb'))
        auc.append(testing(test, model, path=data_path + '/features/'+feat_type+'/', components=comp_num, max_frames=frames))
    
    final_auc = statistics.harmonic_mean(auc)
    print("Harmonic mean of the 3 folds is: ", final_auc)


#cross_val("/home/mpapagatsia/Documents/data_birds/", 'cqt', comp_num=12, frames=431)