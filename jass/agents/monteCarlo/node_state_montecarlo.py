	# HSLU
#
# Created by Pascal Benz on 10/02/2020
#
import logging
import copy
import random
import numpy as np
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.game.const import next_player

    
class NodeState :

    def __init__(self):
        self.currentGame= GameSim(RuleSchieber()) # The Board
        self.playerNo = 0 # the current player
        self.visitCount = 0 # the number of simulations
        self.winScore = 0.0  # the sum of all norm 0->1 scores of the leaves
        self.cardPlayed = 0  # the last card which was played in the game
 
    # all possible Childnodes
    # the next player is either the winner of the previous trick or the next player in the current trick
    # constructs a list of all possible states from current state.
    def getAllPossibleStates(self) -> list():
        # if next player was the winner of the tick, then all cards in his hand are valid
        statelist = []
        # First we get the valid cards of the current player of the GameSim-Object of the MC-Simulation.
        obs = self.currentGame.get_observation()
        valid_cards=np.flatnonzero(self.currentGame.rule.get_valid_cards_from_obs(obs))

        # print('valid cards in node state:{}'.format(valid_cards))
         
        # Then we create the node for the next player by deepcopiing the node of the current player.
        for c in valid_cards:
            state= copy.deepcopy(self)
            # By playing the card, the current player changes for the game
            state.currentGame.action_play_card(c)
            state.cardPlayed = c
            state.playerNo = state.currentGame._state.player
            statelist.append(state)    
            # if we have just finished the trick, then the tricks have to be completed with the just played card
            state.currentGame._state.tricks[state.currentGame._state.nr_tricks-1][3]=c
        # print('inside node state, all possible states trick:{}'.format(statelist[0].getGame()._state.current_trick))       
        return statelist

    def setPlayerNo(self, playerIn: int):
        self.playerNo = playerIn

    # The board on which we will play is the current Game State:
    def setGame(self, gameIn: GameSim):
        self.currentGame = copy.deepcopy(gameIn)

    def getGame(self) -> GameSim:
        return self.currentGame

    def getVisitCount(self):
        return self.visitCount

    def incrementVisit(self):
        self.visitCount += 1

    def addScore(self, score: float):
        self.winScore += score

    def getWinScore(self):
        return self.winScore

    def randomPlay(self): 
        # get a list of all possible positions/cards on the trick and play a random move.
        nextStates=self.getAllPossibleStates()
        randomState = random.choice(nextStates)
        # Here we play one card within the gameObject/Board of the current node
        self.currentGame.action_play_card(randomState.cardPlayed)

    
