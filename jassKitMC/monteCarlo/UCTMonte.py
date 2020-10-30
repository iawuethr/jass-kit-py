# HSLU
#
# Created by Pascal Benz on 10/02/2020
#
import sys
# sys.path.insert(0,"C:/Studium_HSLU/HS_2020/DLFORGAMES/JASSKIT/JASSKIT")
import logging
import copy
import numpy as np
import math
from monteCarlo.node_montecarlo import Node


# https://www.baeldung.com/java-monte-carlo-tree-search

class UCTMonte:
    
    @staticmethod
    def uctValue(totalVisit: int, nodeWinScore: float, nodeVisit: int):
        if (nodeVisit == 0):
            return sys.maxsize
        return (float(nodeWinScore)/float(nodeVisit)) + 1.41 * math.sqrt(math.log(totalVisit) / float(nodeVisit))
    
    @staticmethod
    def findBestNodeWithUCT(node: Node):
        parentVisit = node.getState().getVisitCount()
        maxnode = node.getChildArray()[0]
        for c in node.getChildArray():
            olduct=UCTMonte.uctValue(parentVisit, maxnode.getState().getWinScore(), maxnode.getState().getVisitCount())
            newuct=UCTMonte.uctValue(parentVisit, c.getState().getWinScore(), c.getState().getVisitCount())
            if newuct > olduct :
                maxnode = c
        return maxnode