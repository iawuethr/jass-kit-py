# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import copy
import array
# https://www.journaldev.com/33185/python-add-to-array
from jass.agents.monteCarlo.montecarlo_incomplete import MonteCarloTreeSearchIncomplete
import numpy as np
from jass.agents.monteCarlo.node_montecarlo import Node
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, MAX_PLAYER, next_player, partner_player
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber

# AI-modules
# Necessary to use the ML-model
import pandas as pd
import pickle
import os



class AgentMonteCarloAIIncomplete (Agent):
    """
    Randomly select actions for the match of jass (Schieber)
    """
    def __init__(self):
        # log actions
        self._logger = logging.getLogger(__name__)
        # self._logger.setLevel(logging.INFO)
        # Use rule object to determine valid actions
        self._rule = RuleSchieber()
        # init random number generator
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:

        trumpValues = {'DIAMONDS': 0, 'HEARTS': 1,'SPADES': 2, 'CLUBS': 3, 'OBE_ABE': 4, 'UNE_UFE': 5}
    
        self._logger.info('Trump request')
        # The trained ML needs a 2D-array with categories and cards.
        # first we create the Data-Frame
        columns= card_strings
        columns=np.append(columns,'FH')
        print("category of columns {}".format(columns))
        # Es gibt die Kategories und die Kartenwerte

        dataIn= copy.deepcopy(obs.hand)

        if obs.forehand == -1:
            dataIn=np.append(dataIn,1)
        else: dataIn=np.append(dataIn,0)

        # wir transponieren hier das dataIn
        dataIn=dataIn.reshape(1,37)

        predictFrameX = pd.DataFrame(data= dataIn, index = [0], columns=columns, dtype = int)
        # The key values are in: card_strings
        print("predict Array {}".format(predictFrameX.head()))
        print("shape PredictX {}".format(predictFrameX.shape))


        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            # trained model with push-option True/False
            loaded_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finalized_model.sav'), 'rb'))
            trump = loaded_model.predict(predictFrameX)
            print('trump forehand is: {}'.format(trump[0]))
            # transform trump into integer-value
            return trumpValues[trump[0]]
        # if not push or forehand, select a trump
        # This means: using a model, where the PUSH-option does not exist, 
        # since this option really does not exist for the player to which the trump-decision is pushed.
        loaded_model = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'finalized_model_pushed.sav'), 'rb'))
        trump = loaded_model.predict(predictFrameX)
        print('trump backhand is: {}'.format(trump[0]))
        return trumpValues[trump[0]]


    # We cheat and access as Player the total GameState
    def setStateObject(self, stat: GameState):
        self.state = stat

    # nextPlayerNr: The number of the player in the game (NORTH, SOUTH, WEST, EAST)
    # nextPlayerPosition: The position of the player in the tick

    def action_play_card(self, obs: GameObservation) -> int:
        try:
            return self.action_play_card_intern(obs)
        except Exception as e:
            print(str(e))
            raise

    def action_play_card_intern(self, obs: GameObservation) -> int:
        monteCarloSimulation = MonteCarloTreeSearchIncomplete()
        # We prepare the mock-game for the Monte-Carlo-Simulation
        simulatedGamePre = GameSim(self._rule)
        simulatedGame = GameSim(self._rule)

        # We initialize the Game by the state. The state-information is transferred
        # Basically after every card playing the state has to be updated.
        # Otherwise there would not be a correct shuffling.
        # to the player in arena_play.py
        simulatedGamePre.init_from_state(self.state)

        # The list to save the winner nodes.
        winnerNodesList = []

        # For-loop. Creating Multiple Monte-Carlo-Trees based on
        # reshuffling randomly the cards in the hands of the non-playing-players.
        # --> this is then the random guess (determinism) the playing player makes about
        # the hands of the other players.

        for q in range(0,2):
            # When the action_play_card we play a card. Now we determine,
            # # which player we are (North, South, West, East)
            playerNumber = obs.player
            simulatedGame=self.deterministRandomShuffle(simulatedGamePre)
            # print('Nr. of tricks in Game: {}'.format(simulatedGame._state.nr_tricks))
            # print('simulated Game Pre cards: {})'.format(simulatedGame._state.current_trick))
            #
            # # Starting from the current game state we use the simulatedGame-Object to finish the Game
            # # the simulatedGame-Object returns the card to play which was able to end the game with the highest point Nr.
            finishedNode=monteCarloSimulation.findNextMove(simulatedGame, playerNumber)
            winnerNodesList.append(finishedNode)
            print('visits on a winner node: {}'.format(finishedNode.stat.getVisitCount()))

            # print('finished Game After cards: {})'.format(finishedGame._state.current_trick))

        # Here we determine the winner node with the most visits:
        bestNode=extractWinnerNode(winnerNodesList)
        finishedGame=bestNode.getState().getGame()
        print('most visited node: {}'.format(bestNode.stat.getVisitCount()))

        # if the next move/card to play is still in the same trick, which means that,
        # nr_tricks in the game remain unchanged when the next move is played.
         # then we play the last new entry in the trick of the finishedGame
        if simulatedGame._state.nr_tricks == finishedGame._state.nr_tricks :
            print('Same trick card: {}'.format(finishedGame._state.current_trick[finishedGame._state.nr_cards_in_trick -1]))
            return finishedGame._state.current_trick[finishedGame._state.nr_cards_in_trick -1]
        # if we are in a new trick (--> nr of tricks of finishedGame > simulatedGame)
        # then we play the first card in the trick.
        else:
            # We return the last card of the trick, which has just finished:
            # print('tricks nr simulatedGame: {}'.format(simulatedGame._state.nr_tricks))
            # print('tricks nr finishedGame: {}'.format(finishedGame._state.nr_tricks))
            # print('tricks cards: {}'.format(finishedGame._state.tricks))
            # print('New trick card: {}'.format(finishedGame._state.tricks[simulatedGame._state.nr_tricks][3]))
            return finishedGame._state.tricks[simulatedGame._state.nr_tricks][3]

    def deterministRandomShuffle(self, simPre: GameSim) -> GameSim:
        shuffleGameSim = copy.deepcopy(simPre)
        cardsToShuffle = array.array('i', []) 
        myplayer = int(shuffleGameSim._state.player)

        for n in range(4):
            if n is not  myplayer: 
                cardsToShuffle.extend(copy.deepcopy(np.flatnonzero(shuffleGameSim._state.hands[n,:])))
        # print('length of cards to shuffle: {}'.format(len(cardsToShuffle)))
        np.random.shuffle(cardsToShuffle)

        # Redistribute Cards
        for n in range(4):
            if n is not myplayer: 
                # print('hand of player before redistribution: {}'.format(np.flatnonzero(shuffleGameSim._state.hands[n,:])))
                # Clean the hand
                lengthOfHands = len(np.flatnonzero(shuffleGameSim._state.hands[n,:]))
                shuffleGameSim.state.hands[n,:]=np.zeros(shape=[1, 36], dtype=np.int32)

                # exchange the cards hold by the selected player
                # print('length of hand: {}'.format(lengthOfHands))
                for p in range(0,lengthOfHands):
                    # print('we are here: {}'.format(cardsToShuffle))
                    newCard=cardsToShuffle.pop(0)
                    # set the new card
                    shuffleGameSim._state.hands[n,newCard]=1
        return shuffleGameSim

def extractWinnerNode(winnerNodesList: []) -> []:
    # Here we determine the winner node with the most visits:
    bestNode=Node()
    for l in range(0,len(winnerNodesList)):
        if winnerNodesList[l].stat.visitCount >=  bestNode.stat.visitCount:
            bestNode=winnerNodesList[l]

    return bestNode
    

    