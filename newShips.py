import bones
import math
import time
import random

random.seed(time.time())

class Ship():

    #a ship in this game has several attributes:
    #position
    #velocity
    #money
    #goods
    x = 0.0
    y = 0.0
    xv = 0.0
    yv = 0.0
    money = 0
    goods = 0
    speed = 0.0001

    #probabilities for behaviour
    sailTowardsNearestNodeAndSell = 0.0
    sailTowardsNearestNodeAndBuy = 0.0
    explore = 0.0
    followClosestShip = 0.0
    
    
    #current behaviour
    behaviour = ""

    def __init__(self, shipList, nodeList, startTime):
        self.startTime = startTime
        self.nn = bones.NeuralNet([5, 5, 4, 4, 4])
        self.nn.randomiseWeights()
        self.nn.randomiseBiases()
        self.randomisePosition()
        self.setMoneyGoodsToDefault()

    def reset(self):
        self.randomisePosition()
        self.setMoneyGoodsToDefault()
        self.startTime = time.time()

    def setBehaviour(self):
        behaviourProbabilities = [self.sailTowardsNearestNodeAndSell, self.sailTowardsNearestNodeAndBuy, self.explore, self.followClosestShip]
        greatestProbability = 0
        behaviourName = ""
        count = 0
        for behaviourProbability in behaviourProbabilities:
            if greatestProbability < behaviourProbability:
                greatestProbability = behaviourProbability
                if count == 0:
                    behaviourName = "sailTSell"
                if count == 1:
                    behaviourName = "sailTBuy"
                if count == 2:
                    behaviourName = "explore"
                if count == 3:
                    behaviourName = "followShip"
            count = count + 1

        self.behaviour = behaviourName


    def randomisePosition(self):
        self.x = random.random()
        self.y = random.random()

    def setMoneyGoodsToDefault(self):
        self.money = 1000
        self.goods = 0
        
    def update(self):
        #update input nodes
        self.nn.updateInputNodeValue(0, self.x)
        self.nn.updateInputNodeValue(1, self.y)
        self.nn.updateInputNodeValue(2, self.money)
        self.nn.updateInputNodeValue(3, self.goods)
        self.nn.updateInputNodeValue(4, time.time() - self.startTime)
        #propagate forwards
        self.nn.propagateForward()
        #get output from output nodes and apply to ship
        #probability of behaviours
        #behaviours are:
        self.sailTowardsNearestNodeAndSell = self.nn.layers[-1][0].value
        self.sailTowardsNearestNodeAndBuy = self.nn.layers[-1][1].value
        self.explore = self.nn.layers[-1][2].value
        self.followClosestShip = self.nn.layers[-1][3].value
        self.setBehaviour()
        if self.behaviour == "sailTSell":
            self.sailToClosestNode()
            self.sell()
            
        if self.behaviour == "sailTBuy":
            self.sailToClosestNode()
            self.buy()

        if self.behaviour == "explore":
            bearing = random.random() * 360
            tx = math.cos(math.radians(bearing))
            ty = math.sin(math.radians(bearing))
            self.x = self.x + self.speed * tx
            self.y = self.y + self.speed * ty

        if self.behaviour == "followShip":
            self.sailToClosestShip()


    def sailToClosestShip(self):
        closestShip = self.getClosestShip()
        bearing = getBearingToFrom(closestShip, self)
        tx = math.cos(math.radians(bearing))
        ty = math.sin(math.radians(bearing))
        self.x = self.x + self.speed * tx
        self.y = self.y + self.speed * ty

    def sailToClosestNode(self):
        closestNode = self.getClosestNode()
        bearing = getBearingToFrom(closestNode, self)
        tx = math.cos(math.radians(bearing))
        ty = math.sin(math.radians(bearing))
        self.x = self.x + self.speed * tx
        self.y = self.y + self.speed * ty
        

    def sell(self):
        closestNode = self.getClosestNode()
        if self.distanceToNode(closestNode) < 0.001 and self.goods > 0:
            self.money = self.money + (closestNode.price)
            self.goods = self.goods - 1

    def buy(self):
        closestNode = self.getClosestNode()
        if self.distanceToNode(closestNode) < 0.001 and self.money > closestNode.price:
            self.money = self.money - (closestNode.price)
            self.goods = self.goods + 1
    
    def getClosestNode(self):
        closestNode = 0
        closestDistance = 0
        for node in nodeList:
            distance = pythagoreanTheorum(self, node)
            if closestDistance > distance:
                closestDistance = distance
                closestNode = node
        return node

    def getClosestShip(self):
        closestShip = 0
        closestDistance = 0
        for ship in shipList:
            distance = pythagoreanTheorum(self, ship)
            if closestDistance > distance:
                closestDistance = distance
                closestShip = ship 
        return ship


    def distanceToNode(self, node):
        return pythagoreanTheorum(self, node)

def pythagoreanTheorum(entity1, entity2):
    distance = math.sqrt(((entity1.x - entity2.x) ** 2) + ((entity1.y - entity1.y) ** 2))
    return distance
            
class Node():
    def __init__(self):
        self.price = random.random() * 10
        self.x = random.random()
        self.y = random.random()

def getBearingToFrom(entity1, entity2):
    x = entity1.x
    y = entity1.y
    center_x = entity2.x
    center_y = entity2.y
    angle = math.degrees(math.atan2(y - center_y, x - center_x))
    bearing1 = (angle + 360) % 360
    bearing2 = (90 - angle) % 360
    return bearing1

def findProfit(ship):
    profit = ship.money - 1000
    return profit

nodeList = []
shipList = []

numberOfShips = 20
numberOfNodes = 20

startTime = time.time()

for x in range(numberOfNodes):
    newNode = Node()
    nodeList.append(newNode)

for x in range(numberOfShips):
    newShip = Ship(nodeList, shipList, startTime)
    shipList.append(newShip)

generation = 0

while True:
    time.sleep(1)
    for ship in shipList:
        ship.update()
    if (time.time() - startTime) > (10 * 60):
        shipList.sort(reverse=True, key=findProfit)
        totalProfit = 0
        generation = generation + 1
        for ship in shipList:
            totalProfit = totalProfit + findProfit(ship)
            ship.nn.nextGeneration(shipList[int(random.random() * (len(shipList) / 3))].nn, shipList[int(random.random() * (len(shipList) / 3))].nn)
            ship.reset()
        startTime = time.time()
        print("Generation = ", generation, "Total Profit =", totalProfit)
