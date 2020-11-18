# right now this prints the path and set of actions that the agent chooses for each episode
# usually, the optimal path is the last one it outputs, but still need to determine the definite optimal path
# the graphs show convergence because after a ceratin number of episodes, you can see that the number of time steps increases at a steady rate, showing that each episode is likely the same (i.e. convergence)
# still need to be able to find the optimal path

import numpy as np
import random
import time
import matplotlib.pyplot as plt

class World:
    def __init__(self, kingBool):
        self.gridWidth = 10
        self.gridHeight = 7
        self.startPos = (3,0)
        self.currentPos = self.startPos
        self.goalPos = (3,7)
        self.windVals = [0,0,0,1,1,1,2,2,1,0]
        self.king = kingBool

    def movePosition(self, action): # moves the agent to a new position based on the chosen action, and returns the new position and the reward
        newPos = np.add(list(self.currentPos), list(action))

        windProbability = random.randint(1,3)

        temp = newPos[0] - self.windVals[list(self.currentPos)[1]]

        if(windProbability == 2):
            temp -= 1
        elif(windProbability == 3):
            temp += 2

        # MIGHT NEED TO CHANGE THIS TO SAY THAT IF IT GOES OUTSIDE, PUT IT JUST INSIDE, NOT KEEP IT IN THE SAME
        # if(temp >= 0 and temp < self.gridHeight): # if new position with the wind is off the grid, don't do it
        #     newPos[0] = temp

        if(temp < 0):
            newPos[0] = 0
        elif(temp >= self.gridHeight):
            newPos[0] = self.gridHeight - 1

        if(newPos[0] == list(self.goalPos)[0] and newPos[1] == list(self.goalPos)[1]):
            reward = 1
        else:
            reward = -1

        return [(newPos[0], newPos[1]), reward]

    def allowedActionsFromPos(self, position): # not allowing agent to move off grid, unless because of wind
        allowedMoves = []
        if(position[0] != 0):
            allowedMoves.append((-1,0))
        if(position[0] != self.gridHeight - 1):
            allowedMoves.append((1,0))
        if(position[1] != self.gridWidth - 1):
            allowedMoves.append((0,1))
        if(position[1] != 0):
            allowedMoves.append((0,-1))
        if(self.king): # need to append the proper diagonal actions here. right now, this is a bad way of doing it, but it works
            allowedMoves.append((1,1))
            allowedMoves.append((-1,1))
            allowedMoves.append((1,-1))
            allowedMoves.append((-1,-1))
            if(position == (0,0)): # top left
                allowedMoves.remove((-1,-1))
                allowedMoves.remove((-1,1))
                allowedMoves.remove((1,-1))
            elif(position == (0, self.gridWidth - 1)): # top right (0,9)
                allowedMoves.remove((-1,-1))
                allowedMoves.remove((-1,1))
                allowedMoves.remove((1,1))
            elif(position == (self.gridHeight - 1, 0)): # bottom left (6,0)
                allowedMoves.remove((-1,-1))
                allowedMoves.remove((1, -1))
                allowedMoves.remove((1, 1))
            elif(position == (self.gridHeight - 1, self.gridWidth - 1)): # bottom right (6,9)
                allowedMoves.remove((1,-1))
                allowedMoves.remove((1,1))
                allowedMoves.remove((-1,1))
            else:
                if(position[1] == self.gridWidth - 1):
                    allowedMoves.remove((-1,1))
                    allowedMoves.remove((1,1))
                if(position[1] == 0):
                    allowedMoves.remove((-1,-1))
                    allowedMoves.remove((1,-1))
                if(position[0] == 0):
                    allowedMoves.remove((-1,1))
                    allowedMoves.remove((-1,-1))
                if(position[0] == self.gridHeight - 1):
                    allowedMoves.remove((1,1))
                    allowedMoves.remove((1,-1))

        return allowedMoves


class Agent:
    def __init__(self, world: World):
        #self.actions = [(1,0),(-1,0),(0,1),(0,-1)] # up, down, right, left
        self.alpha = 0.5 # learning rate
        self.epsilon = 0.01 # the percent you want to explore
        self.gamma = 0.9
        self.world = world
        self.Q_values = self.createQTable()
        self.numOfSteps = 0

    def createQTable(self):
        QTable = {} # dictionary where each key is a state that holds a differnt number of possible actions, each which has a q-value
        for x in range(self.world.gridHeight):
            for y in range(self.world.gridWidth):
                position = (x,y)
                QTable[position] = {}
                allowedActions = self.world.allowedActionsFromPos(position)
                for action in allowedActions:
                    QTable[position][action] = 0
        return QTable

    def getBestAction(self, position):
        maxVal = np.NINF
        maxAction = None
        for action, value in self.Q_values[position].items():
            if value > maxVal:
                maxVal = value
                bestAction = action
        return bestAction

    def chooseAction(self, position):
        allowedActions = list(self.Q_values[position].keys()) # go to the dictionary and get the actions allowed in this state
        possibleNextState = self.world.startPos # just to initialize it, we know it will go into loop
        action = None # initializing the return

        randomNum = np.random.rand()
        if(randomNum < self.epsilon):
            subOptimalActionChoiceIndex = random.randint(0, len(allowedActions) - 1)
            action = allowedActions[subOptimalActionChoiceIndex]
        else:
            action = self.getBestAction(position)
        possibleNextState = np.add(list(position), list(action))
        temp = possibleNextState[0] - self.world.windVals[list(possibleNextState)[1]]

        if(temp >= 0):
            possibleNextState[0] = temp

        possibleNextState = (possibleNextState[0], possibleNextState[1])

        return action

    def SARSA(self):
        #currPos = self.world.startPos # start a starting position. currPos is (3,0)
        self.world.currentPos = self.world.startPos
        visited = [] # initalize
        actionsTaken = []
        visited.append(self.world.startPos) # put the first state into it to say we visited it
        chosenAction = self.chooseAction(self.world.currentPos) # choose a random action from (3,0)
        newPosAndReward = None # initialize

        while True: # loop for each step of the episode
            # Take action A, observe R and S'
            newPosAndReward = self.world.movePosition(chosenAction) # move the agent to a new position based on the chosen action
            newPosition = (newPosAndReward[0][0], newPosAndReward[0][1]) # get the reward and new position from that chosen action
            reward = newPosAndReward[1]
            visited.append(newPosition) # say we vsited that state
            actionsTaken.append(chosenAction)
            self.numOfSteps += 1
            # Choose A' from S' using epislon-greedy policy
            nextAction = self.chooseAction(newPosition) # choose an action that is possible from the new position based on the epislon greedy method
            # Update the Q-value of the current position
            self.Q_values[self.world.currentPos][chosenAction] += self.alpha * (reward + self.gamma * self.Q_values[newPosition][nextAction] - self.Q_values[self.world.currentPos][chosenAction]) # perform the update
            # advance to the next state and action
            self.world.currentPos = newPosition # make current position be the new position, and action be the chosen action that we executed
            chosenAction = nextAction

            if(self.world.currentPos == self.world.goalPos): # if the next state is the goal, break
                    break

        print(visited)
        print(actionsTaken)
        #print(self.Q_values.items())

    def Qlearning(self):
        self.world.currentPos = self.world.startPos
        visited = []
        actionsTaken = []
        visited.append(self.world.startPos)
        newPosAndReward = None # initialize

        while True: # loop for each step of the episode
            # Choose A from S using the epislon-greedy policy
            chosenAction = self.chooseAction(self.world.currentPos)
            # Take A and observe R and S'
            newPosAndReward = self.world.movePosition(chosenAction)
            newPosition = newPosAndReward[0]
            reward = newPosAndReward[1]
            visited.append(newPosition) # say we vsited that state
            actionsTaken.append(chosenAction)
            self.numOfSteps += 1
            # find best action from S'
            bestAction = self.getBestAction(newPosition)
            # update current state q-value
            self.Q_values[self.world.currentPos][chosenAction] += self.alpha * (reward + self.gamma * self.Q_values[newPosition][bestAction] - self.Q_values[self.world.currentPos][chosenAction]) # perform the update
            # advance to the next state
            self.world.currentPos = newPosition

            if(self.world.currentPos == self.world.goalPos): # if the next state is the goal, break
                    break

        print(visited)
        print(actionsTaken)


def main():
    # Runs regular SARSA
    testWorld = World(False)
    testAgent = Agent(testWorld)
    timeSteps = [0]
    episodes = [0]
    for x in range(170): # i think this is the number of episodes
        testAgent.SARSA()
        timeSteps.append(testAgent.numOfSteps)
        episodes.append(x)

    plt.plot(timeSteps, episodes)
    plt.suptitle("SARSA")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodes")
    plt.show()

    # Runs regular Q-learning
    testWorld = World(False)
    testAgent = Agent(testWorld)
    timeSteps = [0]
    episodes = [0]
    for x in range(170): # i think this is the number of episodes
        testAgent.Qlearning()
        timeSteps.append(testAgent.numOfSteps)
        episodes.append(x)

    plt.plot(timeSteps, episodes)
    plt.suptitle("Q-learning")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodes")
    plt.show()

    # Runs SARSA with King's Moves
    testWorld = World(True)
    testAgent = Agent(testWorld)
    timeSteps = [0]
    episodes = [0]
    for x in range(170): # i think this is the number of episodes
        testAgent.SARSA()
        timeSteps.append(testAgent.numOfSteps)
        episodes.append(x)

    plt.plot(timeSteps, episodes)
    plt.suptitle("SARSA - King's Moves")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodes")
    plt.show()

    # Runs Q-learning with King's Moves
    testWorld = World(True)
    testAgent = Agent(testWorld)
    timeSteps = [0]
    episodes = [0]
    for x in range(170): # i think this is the number of episodes
        testAgent.Qlearning()
        timeSteps.append(testAgent.numOfSteps)
        episodes.append(x)

    plt.plot(timeSteps, episodes)
    plt.suptitle("Q-learning - King's Moves")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Episodes")
    plt.show()


if __name__ == '__main__':
    main()
