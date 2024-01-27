import math
import random

def roundfour(x):
    if type(x) == float:
        return round(x,4)
    else:
        return [roundfour(i) for i in x]

class TrainingNode:
    def __init__(self,input,output):
        self.input = input
        self.output = output
        self.next = None


class TrainingSet:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0
    
    def add(self,input,output):
        if self.head == None:
            self.head = self.tail = TrainingNode(input,output)
        else:
            self.tail.next = TrainingNode(input,output)
            self.tail = self.tail.next
        self.size += 1


class NeuralNetwork:
    def __init__(self,nblays,szlays,weights,biases):
        self.nblays = nblays
        self.szlays = szlays
        self.weights = weights
        self.biases = biases
        self.dataset = TrainingSet()

    def sigmoid(self,x):
        return 1.0/(1.0+math.e**(-x))
    
    def dsigmoid(self,x):
        return math.e**(-x)/((1.0+math.e**(-x))**2) 
    
    def calculate(self,input):
        return self.calculateAll(input)[self.nblays-1]

    def calculateAll(self,input):
        tmp = self.Z(input)
        out = [[0.0 for j in range(self.szlays[i])] for i in range(self.nblays)]
        for j in range(self.szlays[0]):
            out[0][j] = tmp[0][j]
        for i in range(1,self.nblays):
            for j in range(self.szlays[i]):
                out[i][j] = self.sigmoid(tmp[i][j])
        return out
    
    def Z(self,input):
        if len(input) != self.szlays[0]:
            return -1
        values = [[0.0 for j in range(self.szlays[i])] for i in range(self.nblays)]
        values[0] = input
        if self.nblays >= 2:
            for j in range(self.szlays[1]):
                values[1][j] = self.dot(self.weights[0][j],values[0]) + self.biases[0][j]
        for i in range(2,self.nblays):
            for j in range(self.szlays[i]):
                values[i][j] = self.dot(self.weights[i-1][j],list(map(self.sigmoid,values[i-1])))+self.biases[i-1][j]
        return values
    
    def devs(self,node):
        out = [[0.0 for j in range(self.szlays[i])] for i in range(self.nblays)]
        Z = self.Z(node.input)
        for i in reversed(range(self.nblays)):
            for j in range(self.szlays[i]):
                if i==self.nblays-1:
                    tmp = 2*(self.sigmoid(Z[i][j]) - node.output[j])
                    out[i][j] = tmp
                else:
                    for l in range(self.szlays[i+1]):
                        out[i][j] += self.weights[i][l][j]*self.dsigmoid(Z[i+1][l])*out[i+1][l]
        return out
    
    def addToSet(self,input,output):
        self.dataset.add(input,output)
    
    def lossWithWB(self,weights,biases):
        tmpNN = NeuralNetwork(self.nblays,self.szlays,weights,biases)
        tmpNN.dataset = self.dataset
        return tmpNN.loss()

    
    def loss(self):
        out = 0
        node = self.dataset.head
        for i in range(self.dataset.size):
            tmp = self.norm(self.subtract(self.calculate(node.input),node.output))
            out += tmp**2
            node = node.next
        return out/self.dataset.size
    
    def dloss(self):
        out = [[[[0.0 for k in range(self.szlays[i])] for j in range(self.szlays[i+1])] for i in range(self.nblays-1)],[[0 for j in range(self.szlays[i+1])] for i in range(self.nblays-1)]]
        node = self.dataset.head
        while node != None:
            Z = self.Z(node.input)
            D = self.devs(node)
            for i in reversed(range(self.nblays-1)):
                for j in range(self.szlays[i+1]):
                    for k in range(self.szlays[i]):
                        if i == 0:
                            out[0][i][j][k] += Z[i][k]*self.dsigmoid(Z[i+1][j])*D[i+1][j]/self.dataset.size
                        else:
                            out[0][i][j][k] += self.sigmoid(Z[i][k])*self.dsigmoid(Z[i+1][j])*D[i+1][j]/self.dataset.size
                    out[1][i][j] += self.dsigmoid(Z[i+1][j])*D[i+1][j]/self.dataset.size
            node = node.next
        return out
    
    def add(self,x,y):
        if type(x) == float or type(x) == int:
            return float(x)+float(y)
        else:
            return [self.add(x[i],y[i]) for i in range(len(x))]
    
    def subtract(self,x,y):
        return self.add(x,self.multiply(-1.0,y))
    
    def multiply(self,k,x):
        if type(x) == float or type(x) == int:
            return float(k)*float(x)
        else:
            return [self.multiply(k,x[i]) for i in range(len(x))]
    
    def armijo(self,dx):
        l = 0
        while self.lossWithWB(self.subtract(self.weights,self.multiply(0.5**l,dx[0])),self.subtract(self.biases,self.multiply(0.5**l,dx[1]))) > self.loss() + (0.5**(l+1))*((self.norm(dx))**2):
            l += 1
            print(self.weights,dx)
            print(self.lossWithWB(self.subtract(self.weights,self.multiply(0.5**l,dx[0])),self.subtract(self.biases,self.multiply(0.5**l,dx[1]))), self.loss() + (0.5**(l+1))*((self.norm(dx))**2))
        print(l)
        return 0.5**l
    
    def dotRec(self,x,y):
        if type(x) == float or type(x) == int:
            return float(x)*float(y)
        else:
            out = 0.0
            for i in range(len(x)):
                out += self.dotRec(x[i],y[i])
            return out
    
    def norm(self,x):
        return math.sqrt(self.dot(x,x))
    
    def dot(self,x,y):
        return self.dotRec(x,y)
    
    def train(self):
        grad = self.dloss()
        norm = self.norm(grad)
        i = 0
        while norm > 2e-6 and i < 100000:
            bl = 10
            self.weights = self.subtract(self.weights,self.multiply(bl,grad[0]))
            self.biases = self.subtract(self.biases,self.multiply(bl,grad[1]))
            grad = self.dloss()
            norm = self.norm(grad)
            i += 1

def CreateNN(arr):
    weights = [[[10*(random.random()-0.5) for k in range(arr[i])] for j in range(arr[i+1])] for i in range(len(arr)-1)]
    biases = [[10*(random.random()-0.5) for j in range(arr[i])] for i in range(1,len(arr))]
    return NeuralNetwork(len(arr),arr,weights,biases)