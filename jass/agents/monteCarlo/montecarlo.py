# HSLU
#
# Created by Pascal Benz on 10/02/2020
# https://www.baeldung.com/java-monte-carlo-tree-search

from jass.agents.monteCarlo.tree import Tree
import time
import copy
import numpy as np
from jass.agents.monteCarlo.UCTMonte import UCTMonte
from jass.game.const import next_player, PUSH, partner_player, NORTH, SOUTH
from jass.game.game_sim import GameSim
from jass.agents.monteCarlo.node_montecarlo import Node

# "Board" corresponds "Trick"
# "opponent" corresponds "maximizingPlayer"

class MonteCarloTreeSearch:

    # Out of the children of the root node (root-GameSim) the best node (GameSim)
    # will be chosen and returned. Out of this GameSim the card to play has to be extracted for 
    # action_play_card in agent_most_colour_Monte_Carlo.py
    def findNextMove(self, game: GameSim, playerNo: int) -> np.array: 
        # define an end time which will act as a terminating condition
        end = time.time()*1000.0 + 5000
 
        tree = Tree()
        # The root of the tree is the first GameSim?
        rootNode = Node()
        rootNode.stat.currentGame = copy.deepcopy(game)
  
        rootNode.parentNode =  None
        tree.setRoot(rootNode)
        rootNode.getState().setGame(game)
        rootNode.childNodes = []
        # Here we add the the child nodes, which are the valid cards of the next player:
        # print('next move tick: {})'.format(rootNode.getState().getGame()._state.current_trick))

        while (time.time()*1000.0 < end): 

            # ***************Exploitation*******************
            # The node with the highest UTC will be chosen and
            # played again and again, since there are obviously good solutions (GameSims) there.
            # When the node is played long enough, his UTC will shrink and shrink, till 
            # another child-node will be the most promising node.

            # In the very beginning, when the root node has not yet any children,
            # the root node will be chosen as the most promising node.

            promisingNode = self.selectPromisingNode(rootNode)

            # print('promising node trick1: {}'.format(promisingNode.getState().getGame()._state.current_trick))
            if (not promisingNode.getState().getGame().is_done()):
                # ****************Expansion**********************
                # child nodes are attached to the most promising leaves/unexplored inner nodes
                # the very first promising node at the start will be the root node itself.
                # these child nodes of the promising node are the childrenchilds of the root node.
                self.expandNode(promisingNode)
            
            # print('promising node trick2: {})'.format(promisingNode.getState().getGame()._state.current_trick))
            # *************Exploration*************
            # as long as the node is not the final trick of a GameSim
            # we will choose a random child out of the children of the 
            # most promising node
            nodeToExplore = promisingNode
            # print('children of promising Node: {})'.format(promisingNode.getChildArray()[0].getState().getGame()._state.current_trick))
            if (len(promisingNode.getChildArray()) > 0):
                nodeToExplore = promisingNode.getRandomChildNode()
            
            # print('node to explore: {})'.format(nodeToExplore.getState().getGame()._state.current_trick))
            # *******************Simulation********************
            # Starting from exploration-node finish now the game
            # by randomly choosing played cards.

            #***************** bis hierhin Ok **************************************************************
            playoutResult = self.simulateRandomPlayout(nodeToExplore,playerNo)
            # We are playing the results back to the promising nodes and the root-nodes
            self.backPropogation(nodeToExplore, playoutResult, playerNo)
            # print('Here4')
        
        # Selection: the child node (next GameSim) with the highest UCB1 score is chosen
        # and the card in the real GameSim played. (--> we have to extract the card out of the Game)

        # Check if childnodes contain reasonable results:
        # print('root node again: {}'.format(rootNode.getState().getGame()._state.current_trick))
        
        #for q in rootNode.childNodes :
        #    print('childNode trick: {}'.format(q.getState().getGame()._state.current_trick))

        winnerNode = rootNode.getChildWithMaxScore()
        print('winnerScore: {}'.format(winnerNode.stat.getWinScore()))
        tree.setRoot(winnerNode)
        # print('return Winnercard')
        return winnerNode.getState().getGame()

    def selectPromisingNode(self,rootNode: Node) -> Node: 
        node = rootNode
        # We select the most promising child of the rootnode.
        # If the rootnode has not yet a 
        # print('ChildArrayLength:{}'.format(len(rootNode.getChildArray())))
        # As long as we don't hit a leaf, we traverse down the tree (while-loop)
        # choosing the most promising nodes as path. 
        # Before the first expansion, the promising node will be the rootnode itself.
        while (len(node.getChildArray()) is not 0): 
            node = UCTMonte.findBestNodeWithUCT(node)
        return node

    # This method recommends a leaf node which should be expanded further in the expansion phase:
    def expandNode(self, node: Node):
        possibleStates = node.getState().getAllPossibleStates()
        # print('A trick in a state:{}'.format(possibleStates[0].getGame()._state.current_trick))
        player = 0

        # When the trick has just be won, so the node to explore is [-1,-1,-1,-1] with the 
        # starting player who has just won the tick:
        if node.getState().currentGame._state.nr_cards_in_trick == 0 :
            # the winner of the current trick is the first player of the next trick in the currentGame
            player = node.getState().currentGame._state.trick_winner[node.getState().currentGame._state.nr_tricks]
        # if trick is not yet finished
        else: player = next_player[node.getState().currentGame._state.player] 

        
        

        for c in possibleStates: 
                   newNode = Node()
                   newNode.stat = c
                   newNode.setParent(node)
                   # next player, in case of Jass it depends on the State of the 
                   # game.
                   newNode.getState().setPlayerNo(player)
                   node.getChildArray().append(newNode) 

    def simulateRandomPlayout(self,node: Node, playerNo: int):
        tempNode = copy.deepcopy(node)
        # State of the game played (totally 9 tricks to play)
        tempState = tempNode.getState() 
        gameStatus = tempState.getGame().is_done()

        # As long as the nine ticks of the game are not finished, play the game
        while not gameStatus : 
            tempState.randomPlay()
            gameStatus = tempState.getGame().is_done()

        # We return the number of points, whicht the team of the player gets in the game, after the game is done.
        # Check if the player belong to the team 1
        # The points are weighted by the total score of 157 --> values between 0 and 1
        if playerNo == 0 or partner_player[playerNo] == 0:
            return float(tempState.getGame().state.points[0])/157.0
        else: return float(tempState.getGame().state.points[1])/157.0


    # update function to propagate score and visit count starting from leaf to root
    # in tic-tac-toe playerNo tells us if we have won (=1), draw (0) or lost (-1)
    # in case of a jass, we get different points however.
    def backPropogation(self,nodeToExplore: Node, pointsOfSimulation: float, playerNo: int):
        tempNode = nodeToExplore
        # 157 is the total number of points to win in a game
        while tempNode is not None:
            tempNode.getState().incrementVisit()
            tempNode.getState().addScore(pointsOfSimulation)
            tempNode = tempNode.getParent()
    
