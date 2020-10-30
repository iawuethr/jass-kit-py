# HSLU
#
# Created by Pascal Benz on 10/02/2020
#
import sys
# sys.path.insert(0,"C:/Studium_HSLU/HS_2020/DLFORGAMES/JASSKIT/JASSKIT")
import logging
import copy
import numpy as np
from jassKitMC.monteCarlo.node_montecarlo import Node


# https://www.baeldung.com/java-monte-carlo-tree-search

class Tree:
    def setRoot(self, rootIn: Node):
        self.root= rootIn
    
    def getRoot(self):
        return self.root

    def setRoot(self, nodeIn: Node):
        self.root = nodeIn