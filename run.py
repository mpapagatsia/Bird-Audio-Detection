import sys, getopt
import feature_extraction
import dir_config
import train_svm
import train_rforest

def main(argv):
    features = ''
    classifier = ''
    path = ''

    try:
        opts, args = getopt.getopt(argv,"hf:c:p:",["features=","classifier=", "path="])
    except getopt.GetoptError:
        print ('test.py -f <features> -c <classifier> -p <path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('run.py -f <features> -c <classifier> -p <path>')
            sys.exit()
        elif opt in ("-f", "--features"):
            features = arg
        elif opt in ("-c", "--classifier"):
            classifier = arg
        elif opt in ("-p", "--path"):
            path = arg
            
    if classifier== '' or features == '':
        print("Specify both features and classifier.\n\nrun.py -f <features> -c <classifier> -p <path>")
        sys.exit()

    #configure directories
    if path == '':
        print("Empty path.")
        sys.exit()
    else:
        dir_config.config(path)
    
    feature_extraction.dataset_split(path)

    print ('Features to use: ', features)
    print ('Classifier to use: ', classifier)

    if features == 'opensmile_mfcc':
        feature_extraction.feature_extraction(path, f_type="opensmile_mfcc")
        components = 39
        frames = 998
    elif features == 'opensmile_chroma':
        feature_extraction.feature_extraction(path, f_type="opensmile_chroma")
        components = 12
        frames = 993
    elif features == 'mfcc':
        feature_extraction.feature_extraction(path, f_type="mfcc")
        components = 39
        frames = 431
    elif features == 'cqt':
        feature_extraction.feature_extraction(path, f_type="cqt")
        components = 12
        frames = 431
    elif features == 'cens':
        feature_extraction.feature_extraction(path, f_type="cens")
        components = 12
        frames = 431
    
    if classifier == 'svm':
        train_svm.cross_val(path, features, components, frames)
    elif classifier == 'forest':
        train_rforest.cross_val(path, features, components, frames)
    

if __name__ == "__main__":
    main(sys.argv[1:])
