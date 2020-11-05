from membership import *
from NEFCLASS import *
from data_loading import *
import argparse

def check_accuracy(model, data, targets):
    correct = 0
    total = 0
    for i, (r,t) in enumerate(zip(data, targets)):
        # print(i)
        output = model(r, t)
        pred_class = output.index(max(output))
        if pred_class == t:
            correct +=1
        total += 1
    
    return 100*correct/total

def main(args):
    
    '''
    load dataset
    '''
    if args.dataset == 'iris':
        train_data, train_targets, test_data, test_targets = load_iris(args)
    elif args.dataset == 'bc':
        train_data, train_targets, test_data, test_targets= load_breast_cancer(args)
    elif args.dataset == 'wbc':
        train_data, train_targets, test_data, test_targets= load_breast_cancer_wisconsin(args)
    else:
        print('dataset does not exist')
        assert False
    '''
    define linguistic variable
    '''
    if args.num_sets == 5:
        labels = ['lower','low','average','high','higher']
    elif args.num_sets == 3:
        labels = ['low','average','high']
    elif args.num_sets == 7:
        labels = ['lowest','lower','low','average','high','higher','highest']
    elif args.num_sets == 9:
        labels = ['extremely low','lowest','lower','low','average','high','higher','highest', 'extremely high']
    else:
        print('only 3/5/7 sets supported')
        assert False

    '''
    initialize model
    '''
    model = NEFCLASS(num_input_units= args.num_input_units, num_fuzzy_sets=args.num_sets, \
                    kmax=args.kmax, output_units=args.output_units)
    abcs = [build_membership_function(train_data[d],labels) for d in range(train_data.shape[1])]
    model.init_fuzzy_sets(abcs)
    
    '''
    start rule learning
    '''
    #learn rule
    for i, (r,t) in enumerate(zip(train_data,train_targets)):
        model.learn_rule(r, t)
    print(f'model learnt {model.get_num_rules()} rules')
    
    '''check accuracy after rule learning'''
    print(f'Accuracy on training set after rule learning: {check_accuracy(model, train_data, train_targets):.2f}%')

    
    '''fuzzy set learning'''
    print('start fuzzy set learning')
    best_acc_epoch_pair = [-1,-1]
    for e in range(args.num_epoch):
        for i, (r,t) in enumerate(zip(train_data,train_targets)):
            output = model(r, t)
            delta = [1 - output[i] if i == t  else 0 - output[i] for i in range(len(output))]
            model.update_fuzzy_sets(args.sigma, delta)
        epoch_acc = check_accuracy(model, train_data, train_targets)
        if epoch_acc > best_acc_epoch_pair[0]:
            best_acc_epoch_pair[0] = epoch_acc
            best_acc_epoch_pair[1] = e
        #early stopping
        if e - best_acc_epoch_pair[1] > 10:
            break
        if e % 5 == 0:
            print(f'Epoch {e}: {epoch_acc}')
            # input()
    print(f'Best accuracy {best_acc_epoch_pair[0]} at epoch {best_acc_epoch_pair[1]}')
    
    print(f'Accuracy on test set after fuzzy set learning: {check_accuracy(model, test_data, test_targets):.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NEFCLASS')
    parser.add_argument('--dataset', default='iris', type=str, help='dataset to load')
    parser.add_argument('--sigma', default=0.01, type=float, help='learning rate')
    parser.add_argument('--num_epoch', default=100, type=int, help='number of epoch for fuzzy set learning')
    parser.add_argument('--num_sets', default=5, type=int, help='number of fuzzy sets')
    parser.add_argument('--kmax', default=50, type=int, help='number of fuzzy sets')
    
    args = parser.parse_args()
    main(args)
