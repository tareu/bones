import sys
import bones
import time
import math
import random
import pygame
from pygame.locals import *

random.seed(time.time())
pygame.init()

fps = 30
FramePerSec = pygame.time.Clock()

blue = (0 , 0, 255)
red = (255, 0, 0)
white = (255, 255, 255)
gold = (255, 215, 0)
seablue = (0, 105, 148)

displaysurf = pygame.display.set_mode((600,400))
displaysurf.fill(seablue)
pygame.display.set_caption("ShipGame")
sizeOfPlayArea = 1000
screenLocation = [0,0]
screenSize = [600,400]
class Drawable:
    colour = (0, 0, 0)
    toDraw = False
    position = []
    def __init__(self):
        self.position = [(random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2), (random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2)]
        self.toDraw = False
        print(self.position)
    def update(self):
        self.toDraw = False
        if (self.position[0] > screenLocation[0]) and (self.position[0] < (screenSize[0] + screenLocation[0])):
            if (self.position[1] > screenLocation[1]) and (self.position[1] < (screenSize[1] + screenLocation[1])):
                self.toDraw = True 
    def draw(self):
        pygame.draw.rect(displaysurf,(self.colour),(self.position[0] - screenLocation[0], self.position[1] - screenLocation[1],10, 10))

class Ship(Drawable):
    heading = 0
    nn = 0
    speed = 1
    supplies = 100
    currentAction = "anchor"
    goods = 0
    money = 0
    sightDistance = 200
    sailingProbability = 0
    def __init__(self):
        # neural net input =
        #0b14 inputs  = 127 * number of entity classes for 'objects within sight' 127 is basically a bearing
        #0b16 =  cost of goods of port within trading distance
        #0b16 =  money on boat
        #0b16 =  goods on boat
        #0b16 =  supplies on boat
        self.nn = bones.NeuralNet([14+16+16+16+16, 16, 16, 32, 6, 12])
        self.nn.randomiseWeights()
        self.nn.randomiseBiases()
        self.goods = random.random()*100
        self.money = random.random()*100
        Drawable.__init__(self)
        self.colour = white
    def updateNNInputs(self):
        #we clear all previous inputs
        for x in range(len(nn.layers[0])):
            updateInputNodeValue(x, 0)

        # we iterate through all the ships
        # and add them to our input
        for ship in shipList:
            # check their distance to us (this ship)
            if pythagoreanTheorem(ship, self) < self.sightDistance:
                #so if they are within sight distance
                #we get the bearing, convert it to our strange 7 bit value
                #and update the relevant input nodes
                bearing = getBearing(ship, self)
                bearingBitValue = bearing/2.83464566929
                self.nn.updateInputNodeValue(bearingBitValue, 1)

        nearbyPorts = []
        # we iterate through all the ports
        # and add them to our input
        for port in portList:
            # check their distance to us (this ship)
            distToPort = pythagoreanTheorem(port, self)
            if disToPort < 40:
                nearbyPorts.append(port)
            if distToPort < self.sightDistance:
                #so if they are within sight distance
                #we get the bearing, convert it to our strange 7 bit value
                #and update the relevant input nodes
                bearing = getBearing(port, self)
                # here we add 127 because ports use a different set of 127 inputs
                bearingBitValue = (bearing/2.83464566929) + 127
                self.nn.updateInputNodeValue(bearingBitValue, 1)
        #here we find the cost of goods in the port we are near,
        #doing nothing if not near a port, and converting it to bits
        #then using the next 16 inputs of the nn to place the converted
        #binary number in
        for port in nearbyPorts:
            for bit in convertIntTo16BitBinaryArray(port.goodsPrice):
                for inputNode in range(254,270):
                    self.nn.updateInputNodeValue(inputNode, int(bit))
        for bit in convertIntTo16BitBinaryArray(self.money):
            for inputNode in range(270,270+16):
                self.nn.updateInputNodeValue(inputNode, int(bit))
        for bit in convertIntTo16BitBinaryArray(self.goods):
            for inputNode in range(270+16,270+16+16):
                self.nn.updateInputNodeValue(inputNode, int(bit))
        for bit in convertIntTo16BitBinaryArray(self.supplies):
            for inputNode in range(270+16+16,270+16+16+16):
                self.nn.updateInputNodeValue(inputNode, int(bit))
    def getOutputFromNeuralNet(self):
        count = 0
        for outputNode in self.nn.layers[-1]
            # get the potential bearing of the ship by reading the first 7 bits of 
            # output and converting it into a mod127 number, converting that number
            # into a mod360 number and setting the self.heading value as such
            tempbearing = "" 
            if count < 7:
                if outputNode.value > 0.5:
                    tempbearing += '1'
                if outpuNode.value < 0.5:
                    tempbearing += '0'
            if count == 7:
                self.bearing = int(tempbearing, 2) * 2.83464566929
                #use bit 8 (count 7) to determine whether we sail or not by just putting
                # the value into the property. At the end of this function we will
                # choose what the current state should be by using the highest value.
                self.sailingProbability = outputNode.value
            if count == 8:
                #use bit 9 to determine whether to try to dock or not using the same
                #method that we did for sailing
                self.dockingProbability = outputNode.value
            count = count + 1
                                
    def anchor(self):
        self.supplies = supplies - 1
        pass
    def sail(self, bearing):
        #the get output of nn function formats our bearing into 360 degree format
        self.supplies = supplies - 1
        x = math.cos(math.radians(bearing))
        y = math.sin(math.radians(bearing))
        self.position[0] = self.position[0] + (speed * x)
        self.position[1] = self.position[1] + (speed * y)
    def dock(self):
        for port in portList:
            if pythagoreanTheorem(self, port) < 40:
                self.supplies = self.supplies + 1
                break
    def buyGoods(self, amount):
        for port in portList:
            if pythagoreanTheorem(self, port) < 40:
                if amount < self.money:
                    port.buyGoods(self, amount) 
                break

    def sellGoods(self):
        for port in portList:
            if pythagoreanTheorem(self, port) < 40:
                if amount < self.goods:
                    port.sellGoods(self, amount) 
                break

def convertIntTo16BitBinaryArray(number):
    binaryArray = format(number, "016b")
    return binaryArray

def getBearing(entity1, entity2):                                    
    # note: entity 2 is the centre, the bearing is measured from here
    x - entity1.position[0]
    y - entity1.position[1]
    center_x - entity2.position[0]
    center_y - entity2.position[1]

    angle = math.degrees(math.atan2(y - center_y, x - center_x))
    bearing1 = (angle + 360) % 360
    bearing2 = (90 - angle) % 360
    return bearing1

def pythagoreanTheorem(x1, y1, x2, y2):
    distance = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
    return distance

def pythagoreanTheorem(entity1, entity2):
    distance = math.sqrt(((entity1.position[0] - entity2.position[0])**2) + ((self.position[1] - self.position[1])**2))
    return distance

class Port(Drawable):
    money = 0
    goods = 0
    goodsPrice = 0 
    def __init__(self):
        Drawable.__init__(self)
        self.colour = gold
        self.money = random.random() * 1000
        self.goods = random.random() * 1000
        self.goodsPrice = random.random() * 10
    def sellGoods(self, ship, amount):
        if (amount * self.goodsPrice) < self.money:
            self.goods = self.goods + amount
            self.money = self.money - (amount * self.goodsPrice)
            ship.goods = ship.goods - amount
            ship.money = ship.money + (amount * self.goodsPrice)
    def buyGoods(self, ship, amount):
        if (amount/self.goodsPrice) < self.goods:
            self.goods = self.goods - amount/self.goodsPrice
            ship.goods = ship.goods + amount/self.goodsPrice
            ship.money = ship.money - amount
            self.money = self.moeny + amount

portList = []
for x in range(10):
    p = Port()
    portList.append(p)

shipList = []
for x in range(10):
    s = Ship()
    shipList.append(s)
        
# game loop begins

while True:
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        screenLocation[1] = screenLocation[1] - 5
    if keys[pygame.K_DOWN]:
        screenLocation[1] = screenLocation[1] + 5
    if keys[pygame.K_RIGHT]:
        screenLocation[0] = screenLocation[0] + 5
    if keys[pygame.K_LEFT]:
        screenLocation[0] = screenLocation[0] - 5
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
    displaysurf.fill(seablue)
    for port in portList:
        port.update()
        if port.toDraw:
            port.draw()
    for ship in shipList:
        ship.update()
        if ship.toDraw:
            ship.draw()
    FramePerSec.tick(fps)
    pygame.display.update()
