import sys
import copy
import bones
import time
import math
import random
import pygame
from pygame.locals import *

# nn output is 28 bits:
#                               - we can change this to 7
#                       0       turn left and right (2)
#                       1       sailingProbability
#                       2       dockingProbability
#                       3       sellProbability
#                       4       buyProbability
#                       5       anchorProbability
#                       so 6 bits 1-indexed
#                       
# nn input is ? bits
#                       0-15    ship radar
#                       16-31   port radar
#                       32-47   port prices
#                       48-63   money
#                       64-79   goods
#                       80-95   supplies
#                       96-111  time


# TODO real evolutionary computing
#                       this involves things which I do not fully understand
#                       the example given is the youtube video shows that
#                       given a specific goal, you can select all 
#                       nns which perform that goal best and combine their
#                       characteristics with a little mutation.
#                       the example gives five things needed for
#                       evolution.
#                       1. self-replication  -  this can be done by 
#                                               copying the nn
#                       2. blueprint         -  
#                       3. inherit blueprint
#                       4. mutation
#                       5. selection
blue = (0 , 0, 255)
red = (255, 0, 0)
green = (78, 53, 36)
white = (255, 255, 255)
gold = (255, 215, 0)
seablue = (0, 105, 148)

                      
                       
random.seed(time.time())
pygame.init()

fps = 70 
FramePerSec = pygame.time.Clock()

font = pygame.font.Font('freesansbold.ttf', 12)
def displayText(text, position):
    text = font.render(text, True, green, blue)
    textRect = text.get_rect()
    textRect.center = (position[0], position[1])
    txtInfo = [text, textRect]
    return txtInfo

txtList = []

screenSize = [1200,700]
numberOfShips = 25 
numberOfPorts = 40
sizeOfPlayArea = 300
simTime = 700

displaysurf = pygame.display.set_mode((screenSize[0], screenSize[1]))
displaysurf.fill(seablue)
pygame.display.set_caption("ShipGame")
screenLocation = [0-(screenSize[0]/2),0-(screenSize[1]/2)]
currTicks = pygame.time.get_ticks()

class Drawable:
    colour = (0, 0, 0)
    toDraw = False
    position = []
    def __init__(self):
        self.position = [(random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2), (random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2)]
        self.toDraw = False
    def update(self):
        self.toDraw = False
        if (self.position[0] > screenLocation[0]) and (self.position[0] < (screenSize[0] + screenLocation[0])):
            if (self.position[1] > screenLocation[1]) and (self.position[1] < (screenSize[1] + screenLocation[1])):
                self.toDraw = True 
    def draw(self):
        pygame.draw.rect(displaysurf,(self.colour),((self.position[0] - screenLocation[0])-(self.size/2), (self.position[1] - screenLocation[1])-(self.size/2),self.size, self.size))

class Ship(Drawable):
    # so, possibly how this works is:
    # ship gets instantiated by __init__, we then use the ships'
    # 'update nn inputs' function to update the nn inputs with the latest
    # data, then 'propagateForwards' the nn to get the latest output, then ship's
    # get nn output function to update what the ship does. 
    # that should all be contained in the self.nnStep() function
    size = 2
    name = ""
    heading = 0
    nn = 0
    speed = 1
    supplies = 10000
    goods = 0
    money = 100
    sellAmount = 1
    buyAmount = 1
    sightDistance = 40 
    sailingProbability = 0
    sellProbability = 0
    dockingProbability = 0
    buyProbability = 0
    anchorProbability = 0
    startTime = 0

    def __init__(self):
        # neural net input =
        #0b14 inputs  = 127 * number of entity classes for 'objects within sight' 127 is basically a bearing
        #0b16 =  cost of goods of port within trading distance
        #0b16 =  money on boat
        #0b16 =  goods on boat
        #0b16 =  supplies on boat
        #ob16 = time elapsed
        self.startTime = pygame.time.get_ticks()
        #here, output and input refer to the ship, the ship gets input and gives output... not the neural net. 
        self.noOfOutputBits = 16+16+16+16+16+16+16
        self.noOfInputBits = 6
        self.nn = bones.NeuralNet([self.noOfOutputBits, 59, 32, 19, 12, self.noOfInputBits])
        self.nn.randomiseWeights()
        self.nn.randomiseBiases()
        self.goods = 0
        self.money = 100
        Drawable.__init__(self)
        self.colour = white
    def reset(bestShip, self):
        self.startTime = pygame.time.get_ticks()
        self.money = 100
        self.goods = 0
        self.supplies = 10000
        self.nn.nextGeneration(bestShip.nn)

    def turnPort(self):
        self.heading = (self.heading - 1) % 360
    def turnStarboard(self):
        self.heading = (self.heading + 1) % 360
        
    def nnStep(self):
        self.updateNNInputs()
        self.nn.propagateForward()
        self.getOutputFromNeuralNet()
        self.chooseActivity()
    def updateNNInputs(self):
        outputBits = [0] * (self.noOfOutputBits) 
        #we clear all previous inputs
        for x in range(len(self.nn.layers[0])):
            self.nn.updateInputNodeValue(x, 0)

        # we iterate through all the ships
        # and add them to our input
        for ship in shipList:
            # check their distance to us (this ship)
            if pythagoreanTheorem(ship, self) < self.sightDistance:
                #so if they are within sight distance
                #we get the bearing, convert it to our strange 7 bit value
                #and update the relevant input nodes
                bearing = getBearing(ship, self)
                bearingIn16 = int(bearing/22.5)
                
                outputBits[bearingIn16] = 1

        nearbyPorts = []
        # we iterate through all the ports
        # and add them to our input
        for port in portList:
            # check their distance to us (this ship)
            distanceToPort = pythagoreanTheorem(port, self)
            if distanceToPort < 40:
                nearbyPorts.append(port)
            if distanceToPort < self.sightDistance:
                #so if they are within sight distance
                #we get the bearing, convert it to our strange 7 bit value
                #and update the relevant input nodes
                bearing = getBearing(port, self)
                # here we add 127 because ports use a different set of 127 inputs
                bearingIn16 = int(bearing/22.5)
                #print("bearing to port:", bearingIn127)
                lengthOfShipScanArray = 16
                outputBits[bearingIn16 + lengthOfShipScanArray] = 1
                
        #here we find the cost of goods in the port we are near,
        #doing nothing if not near a port, and converting it to bits
        #then using the next 16 inputs of the nn to place the converted
        #binary number in
        for port in nearbyPorts:
                bitArrayOfPortPrice = convertIntTo16BitBinaryArray(port.goodsPrice)
                for whichBitWeTalkingAbout in range(16):
                    outputBits[32 + whichBitWeTalkingAbout] = bitArrayOfPortPrice[whichBitWeTalkingAbout] 

        moneyBitArray = convertIntTo16BitBinaryArray(self.money)
        for bitWeTalkingAbout in range(16):
            outputBits[48+bitWeTalkingAbout] = moneyBitArray[bitWeTalkingAbout]

        goodsBitArray = convertIntTo16BitBinaryArray(self.goods)
        for bitWeTalkingAbout in range(16):
            outputBits[64+bitWeTalkingAbout] = goodsBitArray[bitWeTalkingAbout]

        suppliesBitArray = convertIntTo16BitBinaryArray(self.supplies)
        for bitWeTalkingAbout in range(16):
            outputBits[80+bitWeTalkingAbout] = suppliesBitArray[bitWeTalkingAbout]
        timeBitArray = convertIntTo16BitBinaryArray((currTicks - self.startTime)/60000)
        for bitWeTalkingAbout in range(16):
            outputBits[96+bitWeTalkingAbout] = timeBitArray[bitWeTalkingAbout]

        for bit in range(len(outputBits)):
            self.nn.updateInputNodeValue(bit, int(outputBits[bit]))


    def getOutputFromNeuralNet(self):
        count = 0
        tempbearing = "" 
        for outputNode in self.nn.layers[-1]:

            if count == 0:

                #if self.name == 0:
                    #print()
                #print(self.name, "port/starboard value:", outputNode.value)

                if outputNode.value > 0.8:
                    self.turnPort()
                
                if outputNode.value < 0.2:
                    self.turnStarboard()

            if count == 1:
                self.sailingProbability = outputNode.value

            if count == 2:
                #use bit 9 to determine whether to try to dock or not using the same
                #method that we did for sailing
                self.dockingProbability = outputNode.value

            if count == 3:
                self.sellProbability = outputNode.value
            if count == 4:
                self.buyProbability = outputNode.value
            if count == 5:
                self.anchorProbability = outputNode.value
            # here we take the binary values of the output ( 8 bits each ) and convert
            # them into amounts to use in our functions (buy, sell)
            count = count + 1
        #print("sellam", self.sellAmount, "buyam", self.buyAmount, "buyp", self.buyProbability, "sellp", self.sellProbability, "sailp", self.sailingProbability, "anchorp", self.anchorProbability, "heading",self.heading) 
    def chooseActivity(self):
        probabilityList = [self.sailingProbability, self.dockingProbability, self.sellProbability, self.buyProbability, self.anchorProbability]
        count = 0
        highest = 0
        highestIndex = 0
        for item in probabilityList:
            if item > highest:
                highest = item
                highestIndex = count
            count = count + 1
        if highestIndex == 0:
            self.sail()
        if highestIndex == 1:
            self.dock()
        if highestIndex == 2:
            self.sellGoods()
        if highestIndex == 3:
            self.buyGoods()
        if highestIndex == 4:
            self.anchor()
                                
    def anchor(self):
        if self.supplies > 0:
            self.supplies = self.supplies - 1
        pass
    def sail(self):
        #the get output of nn function formats our bearing into 360 degree format
        # beware: both heading and bearing are used interchangably...

        #you can only sail if you have supplies
        if self.supplies > 0:

            self.supplies = self.supplies - 1
            x = math.cos(math.radians(self.heading))
            y = math.sin(math.radians(self.heading))
            self.position[0] = self.position[0] + (self.speed * x)
            self.position[1] = self.position[1] + (self.speed * y)
        if self.supplies <= 0:
            self.anchor()
    def dock(self):
        for port in portList:
            if pythagoreanTheorem(self, port) < 10:
                self.supplies = self.supplies + 2
        self.anchor()    
    def buyGoods(self):
        for port in portList:
            if pythagoreanTheorem(self, port) < 10:
                if (self.buyAmount * port.goodsBuyPrice) < self.money:
                    port.buyGoods(self, self.buyAmount) 
        self.anchor()

    def sellGoods(self):
        for port in portList:
            if pythagoreanTheorem(self, port) < 10:
                if self.sellAmount < self.goods:
                    port.sellGoods(self, self.sellAmount) 
        self.anchor()

def convertIntTo16BitBinaryArray(number):
    if number < 0:
        number = 0
    binaryArray = format(int(number), "016b")
    return binaryArray

def getBearing(entity1, entity2):                                    
    # note: entity 2 is the centre, the bearing is measured from here
    x = entity1.position[0]
    y = entity1.position[1]
    center_x = entity2.position[0]
    center_y = entity2.position[1]

    angle = math.degrees(math.atan2(y - center_y, x - center_x))
    bearing1 = (angle + 360) % 360
    bearing2 = (90 - angle) % 360
    return bearing1

def pythagoreanTheorem(x1, y1, x2, y2):
    distance = math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))
    return distance

def pythagoreanTheorem(entity1, entity2):
    distance = math.sqrt(((entity1.position[0] - entity2.position[0])**2) + ((entity1.position[1] - entity2.position[1])**2))
    return distance

class Port(Drawable):
    size = 20
    money = 0
    goods = 0
    goodsPrice = 0 
    def __init__(self):
        Drawable.__init__(self)
        self.colour = green
        self.money = random.random() * 1000
        self.goods = random.random() * 1000
        self.goodsPrice = random.random() * 10
        self.goodsBuyPrice = self.goodsPrice * 0.9
        #this means the port buys them, the ship is the buyer
    def sellGoods(self, ship, amount):
        if (amount * self.goodsBuyPrice) < self.money:
            self.goods = self.goods + amount
            self.money = self.money - (amount * self.goodsBuyPrice)
            ship.goods = ship.goods - amount
            ship.money = ship.money + (amount * self.goodsBuyPrice)
        #see the comment above, so visa versa here
    def buyGoods(self, ship, amount):
        if (amount/self.goodsPrice) < self.goods:
            self.goods = self.goods - amount/self.goodsPrice
            ship.goods = ship.goods + amount/self.goodsPrice
            ship.money = ship.money - amount
            self.money = self.money + amount

portList = []
for x in range(numberOfPorts):
    p = Port()
    portList.append(p)

shipList = []
for x in range(numberOfShips):
    s = Ship()
    s.name = x
    shipList.append(s)

def nextG(shipList):
    highestScore = 0
    shipScoreList = []
#    txtList = []
    for ship in shipList:
#        textPosition = [100, (int(ship.name) * 15) + 30]
#        txtList.append(displayText("name: " + str(ship.name) + " score: " + str(ship.money + (ship.goods * 5)) + " supplies: " + str(ship.supplies), textPosition))
        if len(shipScoreList) == 0:
            shipScoreList.insert(0, ship)
            continue
        
        for x, scoreListShip in enumerate(shipScoreList):
            if (ship.money + (ship.goods * 5) + ship.supplies) > (scoreListShip.money + (scoreListShip.goods * 5) + ship.supplies):
                shipScoreList.insert(x, ship)
                break
        shipScoreList.append(ship)
        
    
    shipList = []
    for x in range(numberOfShips):
        chosenShip = shipScoreList[int(x % (int(len(shipScoreList) / 3) + 1))]
        newShip = copy.deepcopy(chosenShip)
        newShip.supplies = 10000
        newShip.name = x
        newShip.nn = copy.deepcopy(chosenShip.nn)
        newShip.reset(chosenShip)
        newShip.position = [(random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2), (random.random()*sizeOfPlayArea)-(sizeOfPlayArea/2)]
        shipList.append(newShip)
    for ship in shipList:
        print("no.:", ship.name, "supplies", ship.supplies)
    return shipList, txtList

        
# game loop begins

timer = False 
timeCount = 0
while True:
    currTicks = pygame.time.get_ticks()
    timeCount = timeCount + 1
    if timeCount >= simTime:
        timeCount = 0
        shipList, txtList = nextG(shipList)
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        screenLocation[1] = screenLocation[1] - 10
    if keys[pygame.K_DOWN]:
        screenLocation[1] = screenLocation[1] + 10
    if keys[pygame.K_RIGHT]:
        screenLocation[0] = screenLocation[0] + 10
    if keys[pygame.K_LEFT]:
        screenLocation[0] = screenLocation[0] - 10
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_n:
                shipList, txtList = nextG(shipList)
            if event.key == pygame.K_p:
                count = 0
                for ship in shipList:
                    count = count + 1
                    print(count)
                    print("sellam", ship.sellAmount, "buyam", ship.buyAmount, "buyp", ship.buyProbability, "sellp", ship.sellProbability, "sailp", ship.sailingProbability, "anchorp", ship.anchorProbability, "heading",ship.heading) 
    displaysurf.fill(seablue)
    txtList = []
    for ship in shipList:
        textPosition = [100, (int(ship.name) * 15) + 30]
        txtList.append(displayText("name: " + str(ship.name) + " score: " + str(ship.money + (ship.goods * 5) + ship.supplies) + " supplies: " + str(ship.supplies), textPosition))
    for textInfo in txtList:
        displaysurf.blit(textInfo[0], textInfo[1])
    for port in portList:
        port.update()
        if port.toDraw:
            port.draw()
    for ship in shipList:
        ship.nnStep()
        ship.update()
        if ship.toDraw:
            ship.draw()
        
    FramePerSec.tick(fps)
    pygame.display.update()

