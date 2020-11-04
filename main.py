from membership import *
from NEFCLASS import *
import pandas as pd



def main():
    data = pd.read_csv('data/iris.csv')
    class_mapping = {}
    for c in data['Species'].unique():
        class_mapping[c] = len(class_mapping)
    data['class'] = [class_mapping[c] for c in data['Species']]

    targets = data['class'].to_numpy()
    data = data[['SepalLengthCm','SepalWidthCm','PetalWidthCm']].to_numpy()
    
    model = NEFCLASS(num_input_units=3, num_fuzzy_sets=3, kmax=10, output_units=3)
    abcs = [build_membership_function(data[d],['low','average','high']) for d in range(data.shape[1])]
    model.init_fuzzy_sets(abcs)
    
    #learn rule
    for i, (r,t) in enumerate(zip(data,targets)):
        model.learn_rule(r, t)
    #learn fuzzy set
    for i, (r,t) in enumerate(zip(data,targets)):
        output = model(r, t)
        delta = [1 - output[i] if i == t  else 0 - output[i] for i in range(len(output))]
        model.update_fuzzy_sets(delta)
        input()
    
    
    correct = 0
    for i, (r,t) in enumerate(zip(data,targets)):
        # print(i)
        output = model(r, t)
        pred_class = output.index(max(output))
        if pred_class == t:
            correct +=1
    print(correct)

if __name__ == '__main__':
    main()
