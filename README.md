# Bird-Audio-Detection
This is the code for my Diploma Thesis submitted to the University of Thessaly in partial fulfillment of the requirements for the degree of Electrical and Computer Engineering.

This thesis is a part of Task 3 Bird Audio Detection Challenge from DCASE 2018 (http://dcase.community/challenge2018/task-bird-audio-detection). 
The goal of the task is the generalisation to new conditions.

The approach presented refers to the exploration of new audio features that could represent bird sounds more effectively.
Many features and preprocessing steps were tested and the best results came from the following.

Features:
- Chroma CQT
- Chroma CENS

Preprocessing:
- High-Pass filter
- Normalization

Classifiers:
- Random Forest
- SVM.

Chroma CQT and Chroma CENS features respond perfectly with the SVM classifier resulting in 99.43% and 99.83% AUC respectively.

Configurations and details on the code are presented in the config.txt file.
