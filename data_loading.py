import os
import pandas as pd
import random 
import numpy as  np

def shuffle_idxs(length):
    tmp =  list(range(length))
    random.shuffle(tmp)
    return tmp
    

def load_iris(args, path='data/iris.csv'):
    data = pd.read_csv(path)
    class_mapping = {}
    for c in data['Species'].unique():
        class_mapping[c] = len(class_mapping)    
    data['class'] = [class_mapping[c] for c in data['Species']]
    
    targets = data['class'].to_numpy()
    data = data[['SepalLengthCm','SepalWidthCm','PetalWidthCm']].to_numpy()
    vars(args)['num_input_units'] = data.shape[1]
    vars(args)['output_units'] = len(class_mapping)
    
    if args.cv:
        return data, targets, np.max(data,axis=0), np.min(data,axis=0)
        
    else:
        train_data, train_targets = data[0::2], targets[0::2]
        test_data, test_targets = data[1::2], targets[1::2]
        return train_data, train_targets, test_data, test_targets, np.max(data,axis=0), np.min(data,axis=0)

    
def load_wine(args,path ='data/wine/'):
    feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', \
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
    target_col = ['quality']

    red_data = pd.read_csv(os.path.join(path,'winequality-red.csv'), sep=';')
    white_data = pd.read_csv(os.path.join(path,'winequality-white.csv'),sep=';')
    data = red_data.append(white_data)
    classes = []
    for c in data['quality']:
        if c <= 4: #poor
            classes.append(0)
        elif c <= 6: #average
            classes.append(1)
        else:
            classes.append(2)
    data['class'] = classes
    
    targets = data['class'].to_numpy()
    data = data[feature_cols].to_numpy()
    vars(args)['num_input_units'] = data.shape[1]
    vars(args)['output_units'] = 3 #poor average excellent
    
    if args.cv:
        return data, targets, np.max(data,axis=0), np.min(data,axis=0)
        
    else:
        train_data, train_targets = data[0::2], targets[0::2]
        test_data, test_targets = data[1::2], targets[1::2]
        return train_data, train_targets, test_data, test_targets, np.max(data,axis=0), np.min(data,axis=0)
    
    

def load_breast_cancer(args,path ='data/breast_cancer/'):
    train_data = pd.read_csv(os.path.join(path, 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(path, 'test_data.csv'))
    
    class_mapping = {}
    for c in train_data['Class'].unique():
        class_mapping[c] = len(class_mapping)    
    train_data['class_mapped'] = [class_mapping[c] for c in train_data['Class']]
    test_data['class_mapped'] = [class_mapping[c] for c in test_data['Class']]
    vars(args)['num_input_units'] = train_data.shape[1]
    vars(args)['output_units'] = len(class_mapping)
    
    train_targets = train_data['class_mapped'].to_numpy()
    train_data = train_data.drop(['class_mapped', 'Class'], axis=1).to_numpy()
    test_targets = test_data['class_mapped'].to_numpy()
    test_data = test_data.drop(['class_mapped', 'Class'], axis=1).to_numpy()


    assert train_data.shape[0] == train_targets.shape[0]
    assert test_data.shape[0] == test_targets.shape[0]
    
    vars(args)['num_input_units'] = train_data.shape[1]
    vars(args)['output_units'] = len(class_mapping)
    
    return train_data, train_targets, test_data, test_targets, np.max(train_data,axis=0), np.min(train_data,axis=0)
    
def load_breast_cancer_wisconsin(args,path ='data/breast_cancer_wisconsin/'):
    
    data = pd.read_csv(os.path.join(path, 'breast-cancer-wisconsin.data'),\
                    header=None)
    feature_cols =['Clump Thickness', 'Uniformity of Cell Size', \
     'Uniformity of Cell Shape', 'Marginal Adhesion', \
     'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
      'Normal Nucleoli', 'Mitoses']
    data.columns = ['id', *feature_cols, 'Class']

    class_mapping = {}
    for c in data['Class'].unique():
        class_mapping[c] = len(class_mapping)    
    data['class_mapped'] = [class_mapping[c] for c in data['Class']]
    
    for c in feature_cols:
        data[c] = data[c].replace('?', '-9999999')
        max = data[c].max()

        data[c]= data[c].replace('-9999999', max)
        
    
    data[feature_cols] = data[feature_cols].astype(float) 
    vars(args)['num_input_units'] = data.shape[1]
    vars(args)['output_units'] = len(class_mapping)
    
    if args.cv:
        targets = data['class_mapped'].to_numpy()
        data = data[feature_cols].to_numpy()
        
        return data, targets, np.max(data,axis=0), np.min(data,axis=0)
        
    else:
        train_idxs = random.sample(list(range(data.shape[0])), k=int(9*data.shape[0]/10))
        test_idxs = [x not in train_idxs for x in list(range(data.shape[0])) ]
        train_data, train_targets = data.loc[train_idxs, feature_cols].to_numpy(), data.loc[train_idxs, 'class_mapped'].to_numpy()
        test_data, test_targets = data.loc[test_idxs, feature_cols].to_numpy(), data.loc[test_idxs, 'class_mapped'].to_numpy()

        
        assert train_data.shape[0] == train_targets.shape[0]
        assert test_data.shape[0] == test_targets.shape[0]
        
        return train_data, train_targets, test_data, test_targets, np.max(data,axis=0), np.min(data,axis=0)



if __name__ == '__main__':
    load_breast_cancer()