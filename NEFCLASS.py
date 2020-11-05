from membership import *
import numpy as np
class NEFCLASS:
    def __init__(self, num_input_units, num_fuzzy_sets, kmax, output_units):
        self.input = _input_layer(num_input_units, num_fuzzy_sets)
        self.rule = _rule_layer(kmax, output_units)
        self.output = _output_layer(output_units)
        
        
    def init_fuzzy_sets(self,abcs):
        self.input.init_abcs(abcs)
        
    def __call__(self,x, t):
        m,ante = self.input(x)
        o = self.rule(m)
        c = self.output(o)
        return(c)
        
    def learn_rule(self,x, t):

        m, ante = self.input(x)
        o = self.rule.learn(ante,t)
        # c = self.output(o)
    
    def update_fuzzy_sets(self, sigma, delta):
        for n in self.rule.nodes:
            interm = n.update_fuzzy_set_node(delta)           
            if interm is not None:
                self.input.update_fuzzy_sets(sigma,interm)
    
    def get_num_rules(self):
        return len(self.rule.nodes)
        

class _input_layer:
    def __init__(self,num_input_units, num_fuzzy_sets):
        self.num_fuzzy_sets =  num_fuzzy_sets
        self.num_input_units = num_input_units
        self.abcs = None
        self.last_m = None
        self.last_ante = None
        self.last_input = None
        
    def init_abcs(self,abcs):
        self.abcs = abcs
        
    def __call__(self,x):
        self.last_input = x
        m = []
        for i in range(len(x)): 
            m.append([determine_membership(x[i], v) for k, v in self.abcs[i].items()])
        
        ante = [mem.index(max(mem)) for mem in m]
        
        self.last_m = m
        self.last_ante = ante

        return m, ante
        
    def update_fuzzy_sets(self, sigma, interm):
        error_rule, (j1,j2), mu = interm
        key = list(self.abcs[j1].keys())[j2]
        abc= self.abcs[j1][key]
        delta_b = sigma * error_rule * (abc[2] - abc[0]) * np.sign(self.last_input[j1]- abc[1])
        delta_a = -sigma * error_rule * (abc[2] - abc[0]) + delta_b
        delta_c = sigma * error_rule * (abc[2] - abc[0]) + delta_b
        #update
        abc = [abc[0]+delta_a, abc[1]+delta_b, abc[2]+delta_c]
        self.abcs[j1][key] = abc
            
            

class _rule_layer:
    def __init__(self, kmax, output_units):
        self.kmax = kmax
        self.output_units = output_units
        self.nodes = []
        self.antes = []
        
    def __call__(self, m):
        tally = [[] for i in range(self.output_units)]
        for n in self.nodes:
            tally = n(m, tally)
        return tally
            
        
        
    def learn(self, antecedent, consequent):
        if len(self.nodes) < self.kmax:
            if str(antecedent) not in self.antes:
                self._create_node(antecedent, consequent)
                self.antes.append(str(antecedent))
        
        # print(len(self.nodes), len(self.antes))
                
        
    def _create_node(self, antecedent, consequent):
        self.nodes.append(RuleNode(antecedent, consequent, self.output_units))
    
        
        
        

class RuleNode:
    def __init__(self, antecedent, consequent, output_units):
        self.antecedent = antecedent
        self.consequent = consequent
        self.output_units = output_units
        self.last_activation = None
        self.last_min_activation = None
        #each rule is connected to exactly 1 output (consequent)
    
    def __call__(self, m, tally):
        #min as tnorm
        activations = [m[i][self.antecedent[i]] for i in range(len(self.antecedent))]
        self.last_activation = activations
        min_activation = min(activations)
        self.last_min_activation =min_activation
        tally[self.consequent].append(min_activation)
        return tally
    
    def update_fuzzy_set_node(self, delta):
        if self.last_min_activation > 0:
            error_rule = self.last_min_activation * (1-self.last_min_activation) * (delta[self.consequent])
            j = np.argmin(self.last_activation)
            mu = self.last_activation[j]
            return error_rule, (j,self.antecedent[j]) , mu
        else:
            return None
        
    
    
                


        
class _output_layer:
    def __init__(self, output_units):
        self.output_units =  output_units
        
    def __call__(self,o):
        #o is tally
        #max as t-conorm
        output = [max(node) if len(node) != 0 else 0 for node in o]
        # print(output)
        # total = sum(output)
        # print(total)
        # output = [o/total for o in output]
        return output
        
