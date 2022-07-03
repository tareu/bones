import random
import numpy as np
import time

random.seed(time.time())

def sigmoid(x):
  
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig

class Node:
    def __init__(self, value):
        self.value = value 

def createNode(value):
    n = Node(value)
    return n

class NeuralNet:
    #generation is used in the adjustment of variables in next generations as a factor of spread of new random variable numbers
    generation = 0
    def __init__(self, layerConfiguration):

        # formatting the layer configuration parameters to fit code below which
        # uses a weird list structure to create the data structure.
        self.generation = 1
        layerConfFormatted = []
        for layerNodeCount in layerConfiguration:
            layerNodeCountFormatted = []
            for node in range(layerNodeCount):
                layerNodeCountFormatted.append(0)
            layerConfFormatted.append(layerNodeCountFormatted)
        layerConfiguration = layerConfFormatted

        self.layers = []

        for layer in range(len(layerConfiguration)):
            nodesInLayer = []
            for nodes in range(len(layerConfiguration[layer])):
                nodesInLayer.append(createNode(0))
                if layer == 0:
                    nodesInLayer[-1].value = 0.5
                #print("node value layer =", layer, "node =", nodesInLayer[-1].value)
            self.layers.append(nodesInLayer)

        # self.layers = [[createNode(random.random()) for i in len(layerConfiguration)] for j in len(layerConfiguration[i])]

        self.weights = []
        for layer in range(len(layerConfiguration)):
            weightLayer = []
            self.weights.append(weightLayer)
            for node in range(len(layerConfiguration[layer])):
                weightLayerNode = []
                self.weights[layer].append(weightLayerNode)
                for prevNodes in range(len(layerConfiguration[layer-1])):
                    self.weights[layer][node].append(random.random())
                    

        # self.weights = [[[random.random() for i in len(layerConfiguration)] for j in len(layerConfiguration[i])] for c in len(layerConfiguration[i-1])]

        self.biases = []
        
        for layer in range(len(layerConfiguration)):
            nodesInLayer = []
            for bias in range(len(layerConfiguration[layer])):
                nodesInLayer.append(random.random())
            self.biases.append(nodesInLayer)

        # self.biases = [[0 for i in len(layerConfiguration)] for j in len(layerConfiguration[i])]


    def updateInputNodeValue(self, inputNodeNum, value):
        self.layers[0][inputNodeNum].value = value



    def randomiseWeights(self):
        # for each layer
        for x in range(1, len(self.layers)): 
            # for each node in layer
            for y in range(len(self.layers[x])):
                # for each node in previous layer
                for z in range(len(self.layers[x - 1])):
                    #print(x, y, z)
                    self.weights[x][y][z] = random.random() - 0.5
                    #for next gen, if prev best was -0.3,  random.random/2 - 0.5 + prevbest , which equals new halfed range centred on previous best answer
                    #print("weight layer =", x, "node =", y, "previous node =", z, self.weights[x][y][z])

    def narrowRandomiseWeights(self):
        # for each layer
        for x in range(1, len(self.layers)): 
            # for each node in layer
            for y in range(len(self.layers[x])):
                # for each node in previous layer
                for z in range(len(self.layers[x - 1])):
                    #print(x, y, z)
                    previousBest = self.weights[x][y][z]
                    self.weights[x][y][z] = (random.random() * (0.9 ** self.generation)) - (0.5 * (9 ** self.generation)) + previousBest
    def nextGeneration(self):
        self.generation = self.generation + 1
        narrowRandomiseWeights(self)
        narrowRandomiseBiases(self)

    def randomiseBiases(self):
        # for each layer except first
        for x in range(1, len(self.layers)): 
            # for each node in layer
            for y in range(len(self.layers[x])):
                self.biases[x][y] = (random.random() * 40) - 20
    def narrowRandomiseBiases(self):
        # for each layer except first
        for x in range(1, len(self.layers)): 
            # for each node in layer
            for y in range(len(self.layers[x])):
                previousBest = self.biases[x][y]
                self.biases[x][y] = (random.random() * (40 * (0.9 ** self.generation))) - (((40 * (0.9 ** self.generation)))/2) 
                #here centre is as above. when making nect generation of nn, new bias should 
                #centre on best case random bias and half the area in which random numbers 
                #can come from. for instance if best case random bias is +12 the generated 
                #random numbers should come from the range random.random() * 20 (halving the 
                #range) + 2 (centering range on 12 +-10)

    def printOutput(self):
        for j in range(len(nn.layers)):
            #print("layer =", j)
            for i in range(len(nn.layers[j])):
                if j == len(nn.layers) - 1:
                    print(nn.layers[j][i].value)

    def propagateForward(self):
        # for each layer except first
        for x in range(1, len(self.layers)):
                # for every node in layer
                for y in range(len(self.layers[x])):
                    sumOfWeights = 0
                    # for every node in previous layer
                    for z in range(len(self.layers[x - 1])):
                        # multiply previous node value with weight
                        sumOfWeights = sumOfWeights + (self.weights[x][y][z] * self.layers[x - 1][z].value)
                    sumOfWeights = sumOfWeights + self.biases[x][y]
                    self.layers[x][y].value = sigmoid(sumOfWeights)
#layerConfiguration = [6, 5, 28, 4, 39, 6]
#for x in range(10):
#   print("nn", x)
#   nn = NeuralNet(layerConfiguration)
#   nn.randomiseWeights()
#   nn.randomiseBiases()
#   nn.propagateForward()
#   
#   
#   nn.printOutput()
