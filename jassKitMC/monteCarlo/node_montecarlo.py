# HSLU
#
# Created by Pascal Benz on 10/02/2020
#
import logging
import copy
import numpy as np
from monteCarlo.node_state_montecarlo import NodeState
import random
# https://www.baeldung.com/java-monte-carlo-tree-search

class Node:

    def __init__(self):
        self.stat = NodeState()
        self.childNodes = []
        self.parentNode:  Node


    def setParent(self, node: 'Node'):
        self.parentNode = node

    def getParent(self) -> 'Node':
        return self.parentNode    

    def getState(self) -> NodeState():
        return self.stat

    def getChildArray(self) -> []:
        return self.childNodes
    
    def getRandomChildNode(self) -> 'Node':
        return random.choice(self.childNodes)

    def getChildWithMaxScore(self) -> 'Node':
        maxScoreNode = Node()
        maxScoreNode.stat.winScore = 0

        for c in self.childNodes:
            if c.stat.winScore > maxScoreNode.stat.winScore:
                maxScoreNode = c
        
        return maxScoreNode


    
