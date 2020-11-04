from membership import *
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
        
        

class _input_layer:
    def __init__(self,num_input_units, num_fuzzy_sets):
        self.num_fuzzy_sets =  num_fuzzy_sets
        self.num_input_units = num_input_units
        self.abcs = None
        
    def init_abcs(self,abcs):
        self.abcs = abcs
        
        
    def __call__(self,x):
        m = []
        for i in range(len(x)): 
            m.append([determine_membership(x[i], v) for k, v in self.abcs[i].items()])
        
        ante = [mem.index(max(mem)) for mem in m]
        return m, ante
            

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
        
    def _create_node(self, antecedent, consequent):
        self.nodes.append(RuleNode(antecedent, consequent, self.output_units))
        
        

class RuleNode:
    def __init__(self, antecedent, consequent, output_units):
        self.antecedent = antecedent
        self.consequent = consequent
        self.output_units = output_units
        #each rule is connected to exactly 1 output (consequent)
    
    def __call__(self, m, tally):
        #min as tnorm
        tally[self.consequent] += [min([m[i][self.antecedent[i]] for i in range(len(self.antecedent))])]
        return tally
    
                


        
class _output_layer:
    def __init__(self, output_units):
        self.output_units =  output_units
        
    def __call__(self,o):
        #o is tally
        #max as t-conorm
        output = [max(node) for node in o]
        # print(output)
        # total = sum(output)
        # print(total)
        # output = [o/total for o in output]
        return output
        
