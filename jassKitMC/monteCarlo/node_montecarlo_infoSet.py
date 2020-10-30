# HSLU
#
# Created by Pascal Benz on 10/02/2020
#
import logging
import copy
import numpy as np
from monteCarlo.node_state_montecarlo import NodeState
from monteCarlo.node_state_montecarlo_Info import NodeStateInfo
import random
# https://www.baeldung.com/java-monte-carlo-tree-search

class NodeInfo:

    def __init__(self):
        self.stat = NodeStateInfo()
        self.childNodes = []
        self.parentNode:  NodeInfo


    def setParent(self, node: 'NodeInfo'):
        self.parentNode = node

    def getParent(self) -> 'NodeInfo':
        return self.parentNode    

    def getState(self) -> NodeState():
        return self.stat

    def getChildArray(self) -> []:
        return self.childNodes
    
    def getRandomChildNode(self) -> 'NodeInfo':
        return random.choice(self.childNodes)

    def getChildWithMaxScore(self) -> 'NodeInfo':
        maxScoreNode = NodeInfo()
        maxScoreNode.stat.winScore = 0

        for c in self.childNodes:
            if c.stat.winScore > maxScoreNode.stat.winScore:
                maxScoreNode = c
        
        return maxScoreNode


    
