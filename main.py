import logging
import random
import numpy as np
import matplotlib.pyplot as pl
from operator import itemgetter
import pygame
from copy import deepcopy


CHANCE = 1


def drawSquare(screen, currentColour, currentColumn, cellSize, currentRow):
    pygame.draw.rect(screen, currentColour, [currentColumn * cellSize, currentRow * cellSize, (currentColumn + 1)
                                             * cellSize, (currentRow + 1) * cellSize])


def countcolors(matrix, color):
    k = 0
    for row in matrix:
        for cell in row:
            if cell == color:
                k += 1
    return k


def random_immune(matrix, color='3'):
    for i in range(len(matrix)):
        new_row = ''
        for j in range(len(matrix[i])):
            if matrix[i][j] == color:
                t = random.randrange(1, 10001)
                if t > 9900:
                    print(1)
                    new_row += '0'
                else:
                    print(2)
                    new_row += matrix[i][j]
            else:
                new_row += matrix[i][j]
        matrix[i] = new_row
    return matrix


def drawGenerationUniverse(cellCountX, cellCountY, universeTimeSeries):
    pygame.init()
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    RED = (255, 0, 0)

    screenHeight = 800
    screenWidth = 800

    cellSize = screenHeight / cellCountX

    size = [int(screenHeight), int(screenWidth)]
    screen = pygame.display.set_mode(size)
    screen.fill(WHITE)

    clock = pygame.time.Clock()
    FPS = 60
    playtime = 0
    cycletime = 0
    interval = .1
    picnr = 0
    currentTimeStep = 0
    while simulationIterations:
        pygame.event.get()
        milliseconds = clock.tick(FPS)  # milliseconds passed since last frame
        seconds = milliseconds / 1000.0  # seconds passed since last frame (float)
        playtime += seconds
        cycletime += seconds
        if cycletime > interval:
            if currentTimeStep >= simulationIterations:
                currentTimeStep = 0
                pygame.time.delay(1000)
                pygame.quit()
            else:
                currentTimeStep += 1
            pygame.time.delay(1000)
            pygame.display.set_caption("TimeStep %3i:  " % currentTimeStep)
            picnr += 1
            if picnr > 5:
                picnr = 0
            cycletime = 0

            for currentRow in range(cellCountY):  # Draw a solid rectangle
                for currentColumn in range(cellCountX):
                    if currentTimeStep > 0 and currentTimeStep < simulationIterations:
                        if universeTimeSeries[currentTimeStep][currentRow][currentColumn] == '0':
                            currentColour = BLUE
                        if universeTimeSeries[currentTimeStep][currentRow][currentColumn] == '1':
                            currentColour = YELLOW
                        if universeTimeSeries[currentTimeStep][currentRow][currentColumn] == '2':
                            currentColour = RED
                        if universeTimeSeries[currentTimeStep][currentRow][currentColumn] == '3':
                            currentColour = GREEN
                        if universeTimeSeries[currentTimeStep][currentRow][currentColumn] == '4':
                            currentColour = BLACK
                        drawSquare(screen, currentColour, currentColumn, cellSize, currentRow)
        pygame.display.update()
    return drawGenerationUniverse()


def printGenerationUniverse(currentTimeStep, cellCountX, cellCountY, susceptibleCharacter, \
                            exposedCharacter, infectedCharacter, recoveredCharacter, deadCharacter):
    logging.info("TimeStep %3i:  " % currentTimeStep)
    rowLabel = ""
    for l in range(cellCountX):
        rowLabel += str(l) + " "
    logging.info(rowLabel)
    for currentRow in range(cellCountY):
        print("%s %s" % (currentRow, universeList[currentRow].replace('0', susceptibleCharacter + " ").replace('1',
                                                                                                               exposedCharacter + " ").
                         replace('2', infectedCharacter + " ").replace('3', recoveredCharacter + " ").replace('4',
                                                                                                              deadCharacter + " ")))


def newStateVN(currentRowNeighbours, upperCharacter, lowerCharacter):
    leftCharacter = currentRowNeighbours[0]
    selfCharacter = currentRowNeighbours[1]
    rightCharacter = currentRowNeighbours[2]

    newState = selfCharacter
    if selfCharacter == '3':  # .S->I
        if leftCharacter == '2' or rightCharacter == '2' or upperCharacter == '2' or lowerCharacter == '2':
            Pichance = (1 - np.random.uniform())
            if 0 < Pichance < InfectedChance:
                newState = '2'
            else:
                Pdchance = (1 - np.random.normal(0.5, 1.0))
                if 0 < Pdchance < DeathResistanceChance:
                    newState = '4'

    elif selfCharacter == '2':
        Prchance = (1 - np.random.normal(0.5, 1.0))
        Pschance = (1 - np.random.normal(0.5, 1.0))
        if 0 < Prchance < ResistanceChance:
            newState = '0'
        else:
            if 0 < Pschance < DeathInfectedChance:
                newState = '4'

    elif selfCharacter == '0':
        Pschance = (1 - np.random.normal(0.5, 1.0))
        if 0 < Pschance < LossOfImmunity:
            newState = '3'
        else:
            Pdchance = (1 - np.random.normal(0.5, 1.0))
            if 0 < Pdchance < DeathResistanceChance:
                newState = '4'

    elif selfCharacter == '1':
        if leftCharacter == '3' or rightCharacter == '3' or upperCharacter == '3' or lowerCharacter == '3':
            Pbchance = (1 - np.random.normal(0.5, 1.0))
            if 0 < Pbchance < BirthChance:
                newState = '3'

    elif selfCharacter == '1':
        if leftCharacter == '2' or rightCharacter == '2' or upperCharacter == '2' or lowerCharacter == '2':
            Pichance = (1 - np.random.normal(0.5, 1.0))
            if 0 < Pichance < InfectedChance:
                newState = '2'

    elif selfCharacter == '4':
        if leftCharacter == '0' or rightCharacter == '0' or upperCharacter == '0' or lowerCharacter == '0':
            Plchance = (1 - np.random.normal(0.5, 1.0))
            if 0 < Plchance < BirthChance:
                newState = '1'
    return newState


def getNewState2Ddiff(currentRowNeighbours, upperCharacter, lowerCharacter):
    difchance = 0.1
    leftCharacter = currentRowNeighbours[0]
    selfCharacter = currentRowNeighbours[1]
    rightCharacter = currentRowNeighbours[2]

    q = 'self'

    if selfCharacter == '4':
        return selfCharacter, q

    swap1 = {'self': selfCharacter, 'right': rightCharacter, 'left': leftCharacter, 'lower': lowerCharacter,
             'upper': upperCharacter}

    swap = {}
    for k, v in swap1.items():
        if v != '4' and v != '1' and v != '-':
            swap[k] = v

    if np.random.uniform() > difchance:
        try:
            q = random.choice(list(swap.keys()))
            selfCharacter = swap[q]
        except IndexError:
            pass

    return selfCharacter, q


InfectedChance = 0.5  # beta
ResistanceChance = 0.025  # gamma
LossOfImmunity = 0.001  # alpha
BirthChance = 0.05  # epsilon
DeathResistanceChance = 0.01  # rho
DeathInfectedChance = 0.01  # rho1

# InfectedChance = 0.2
# ResistanceChance = 0.1
# LossOfImmunity = 0.01
# BirthChance = 0.1
# DeathResistanceChance = 0.01
# DeathInfectedChance = 0.001

simulationIterations = 50
cellCountX = 10
cellCountY = 10
hexagonLayout = False

susceptibleCharacter = 'S'
exposedCharacter = 'E'
recoveredCharacter = 'R'
infectedCharacter = 'I'
deadCharacter = 'D'
extremeEndValue = '0'
timeStart = 0.0
timeEnd = simulationIterations
timeStep = 1
timeRange = np.arange(timeStart, timeEnd + timeStart, timeStep)
universeList = []


def centeredInitialization(cellCountX, cellCountY):
    List = ['3' * cellCountX for i in range(cellCountY)]
    temp = ' '
    for i in range(cellCountX):
        if i == cellCountX // 2:
            temp += '2'
        else:
            temp += '3'
    List[cellCountY // 2] = temp
    return List


S = int(input())

if S == 2:
    for currentColumn in range(cellCountY):
        universe = ''.join(random.choice('3333333332') for universeColumn in range(cellCountX))
        universeList.append(universe)
else:
    universeList = centeredInitialization(cellCountX, cellCountY)

InitSusceptibles = 0.0
InitInfected = 0.0
InitRecovered = 0.0
InitVariables = [InitSusceptibles, InitInfected, 0.0, 0.0, 0.0]

RES = [InitVariables]

universeTimeSeries = []

for currentTimeStep in range(simulationIterations):
    flag1 = True
    if currentTimeStep < 0:
        printGenerationUniverse(currentTimeStep, cellCountX, cellCountY, susceptibleCharacter, exposedCharacter,
                                infectedCharacter, recoveredCharacter, deadCharacter)
    zeroCount = 0
    oneCount = 0
    twoCount = 0
    threeCount = 0
    fourCount = 0
    for currentRow in range(cellCountY):
        zeroCount += universeList[currentRow].count('0')
        oneCount += universeList[currentRow].count('1')
        twoCount += universeList[currentRow].count('2')
        threeCount += universeList[currentRow].count('3')
        fourCount += universeList[currentRow].count('4')
    RES.append([zeroCount, oneCount, twoCount, threeCount, fourCount, currentTimeStep])
    universeTimeSeries.append(deepcopy(universeList))
    for currentRow in range(cellCountY):
        newUniverseRow = ''
        for currentColumn in range(cellCountX):
            if currentRow != 0:
                upperRowNeighbours = universeList[currentRow - 1][currentColumn]
            else:
                upperRowNeighbours = '-'
            if currentRow != cellCountY - 1:
                lowerRowNeighbours = universeList[currentRow + 1][currentColumn]
            else:
                lowerRowNeighbours = '-'

            if currentColumn == 0:
                currentRowNeighbours = '-' + universeList[currentRow][currentColumn:currentColumn + 2]
            elif currentColumn == cellCountX - 1:
                currentRowNeighbours = universeList[currentRow][currentColumn - 1:currentColumn + 1] + '-'
            else:
                currentRowNeighbours = universeList[currentRow][currentColumn - 1:currentColumn + 2]
            newState = newStateVN(currentRowNeighbours, upperRowNeighbours, lowerRowNeighbours)
            if currentColumn == 0:
                universeList[currentRow] = newState + universeList[currentRow][currentColumn + 1:]
            elif currentColumn == cellCountX:
                universeList[currentRow] = universeList[currentRow][:currentColumn] + newState
            else:
                universeList[currentRow] = universeList[currentRow][:currentColumn] + newState + universeList[
                                                                                                     currentRow][
                                                                                                 currentColumn + 1:]
            newState, ind = getNewState2Ddiff(currentRowNeighbours, upperRowNeighbours, lowerRowNeighbours)
            a = deepcopy(universeList[currentRow])
            if ind == 'self':
                pass
            elif ind == 'upper':
                universeList[currentRow - 1] = universeList[currentRow - 1][:currentColumn] + currentRowNeighbours[1] + \
                                               universeList[currentRow - 1][currentColumn + 1:]
                universeList[currentRow] = universeList[currentRow][:currentColumn] + newState + universeList[
                                                                                                     currentRow][
                                                                                                 currentColumn + 1:]
            elif ind == 'lower':
                universeList[currentRow + 1] = universeList[currentRow + 1][:currentColumn] + currentRowNeighbours[1] + \
                                               universeList[currentRow + 1][currentColumn + 1:]
                universeList[currentRow] = universeList[currentRow][:currentColumn] + newState + universeList[
                                                                                                     currentRow][
                                                                                                 currentColumn + 1:]
            elif ind == 'right':
                universeList[currentRow] = universeList[currentRow][:currentColumn] + newState + currentRowNeighbours[
                    1] + universeList[currentRow][currentColumn + 2:]
            elif ind == 'left':
                universeList[currentRow] = universeList[currentRow][:currentColumn - 1] + currentRowNeighbours[
                    1] + newState + \
                                           universeList[currentRow][currentColumn + 1:]
            # try:
            #     if (countcolors(universeTimeSeries[currentTimeStep], '3') + countcolors(universeTimeSeries[currentTimeStep],
            #                                                                             '0')) / countcolors(
            #             universeTimeSeries[currentTimeStep], '2') > CHANCE:
            #         random_immune(universeTimeSeries[currentTimeStep])
            # except ZeroDivisionError:
            #     pass
        try:
            if countcolors(
                    universeList, '2') / (countcolors(universeList, '3') + countcolors(universeList,
                                                                                    '0')) > CHANCE:
                universeList = random_immune(universeList)
        except ZeroDivisionError:
            pass

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
RES.pop(0)
# print(*RES, sep='\n')
pl.plot([x[-1] for x in RES], [x[3] for x in RES], 'green', label='Восприимчивые')
pl.plot([x[-1] for x in RES], [x[2] for x in RES], 'red', label='Больные')
pl.plot([x[-1] for x in RES], [x[0] for x in RES], 'blue', label='Выздоровевшие')
pl.legend(loc=0)

pl.xlabel('Time')
pl.ylabel('Population')
pl.show()
drawGenerationUniverse(cellCountX, cellCountY, universeTimeSeries)
