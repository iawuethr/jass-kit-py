# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import copy
import array
# https://www.journaldev.com/33185/python-add-to-array
from monteCarlo.montecarlo_incomplete import MonteCarloTreeSearchIncomplete
import numpy as np
from jassKitMC.monteCarlo.node_montecarlo import Node
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, MAX_PLAYER, next_player, partner_player
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from jass.arena.arena import Arena
import os 

# AI-modules
# Necessary to use the ML-model
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle



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

        dir_path = os.path.dirname(os.path.realpath(__file__))

        if obs.forehand == -1:
            print('directory path: {}'.format(dir_path))
            # if forehand is not yet set, we are the forehand player and can select trump or push
            # trained model with push-option True/False
            loaded_model = pickle.load(open(os.path.join(dir_path, 'finalized_model.sav'), 'rb'))
            trump = loaded_model.predict(predictFrameX)
            print('trump forehand is: {}'.format(trump[0]))
            # transform trump into integer-value
            return trumpValues[trump[0]]
        # if not push or forehand, select a trump
        # This means: using a model, where the PUSH-option does not exist, 
        # since this option really does not exist for the player to which the trump-decision is pushed.
        loaded_model = pickle.load(open(os.path.join(dir_path, 'finalized_model_pushed.sav'), 'rb'))
        trump = loaded_model.predict(predictFrameX)
        print('trump backhand is: {}'.format(trump[0]))
        return trumpValues[trump[0]]

    # nextPlayerNr: The number of the player in the game (NORTH, SOUTH, WEST, EAST)
    # nextPlayerPosition: The position of the player in the tick

    def action_play_card(self, obs: GameObservation) -> int:

        card_array=[]
        for n in range(36):
            card_array.append(n)

        monteCarloSimulation = MonteCarloTreeSearchIncomplete()
        # We prepare the mock-game for the Monte-Carlo-Simulation
        simulatedGamePre = GameSim(self._rule)
        simulatedGame = GameSim(self._rule)

        # We initialize the Game by the state. The state-information is transferred 
        # Basically after every card playing the state has to be updated.
        # Otherwise there would not be a correct shuffling.
        # to the player in arena_play.py
        # simulatedGamePre.init_from_state(self.state)

        handsIn = np.zeros(shape=[4, 36], dtype=np.int32)
        currentPlayer = obs.player_view
        # print('dealer: {} '.format(obs.dealer))
        # print('current player: {} '.format(currentPlayer))
        #the current trick on the table
        currentTrick =  obs.current_trick
        oldTricks = obs.tricks
        # print('current trick: {} '.format(currentTrick))
        handPlayer=np.flatnonzero(obs.hand)
        # print('hand Player: {} '.format(handPlayer))

        # Remove cards of player in card_array
        # print('number of tricks: {} '.format(obs.nr_tricks))
        # print('card array second: {} '.format(card_array))
        # print('handPlayer second: {} '.format(handPlayer))
        # print('old tricks second: {} '.format(oldTricks[1,:]))

        if obs.nr_tricks > 0:
            print('here we are')
            card_array=list(set(card_array) - set(handPlayer)-set(currentTrick))
            for m in range(obs.nr_tricks+1):
                card_array= list(set(card_array) - set(oldTricks[m,:]))
        else: card_array=list(set(card_array) - set(handPlayer)-set(currentTrick))

        # print('remaining Cards to shuffle: {} '.format(card_array))
        # print('len current trick: {} '.format(len(currentTrick)))
        

        for c in handPlayer:
            handsIn[currentPlayer,c] = 1
        
        # print('handPlayer Zero: {} '.format(handsIn))
        
        playerTypes = [0,1,2,3]
        # print('playerTypes before: {} '.format(playerTypes))
        playerTypes.remove(currentPlayer)
        # print('playerTypes: {} '.format(playerTypes))

        cardsTodistribute=copy.deepcopy(card_array)   
        currentPlayerHandsize= len(handPlayer)
        nextPlayer= next_player.index(currentPlayer)

        playerfirst=nextPlayer
        playersecond=next_player.index(playerfirst)
        playerthird=next_player.index(playersecond)

        cardfirst=[]
        cardsecond=[]
        cardthird=[]

        if obs.nr_cards_in_trick is 0:
            newarr = np.array_split(cardsTodistribute, 3)
            cardfirst= newarr[0]
            cardsecond= newarr[1]
            cardthird= newarr[2]

        if obs.nr_cards_in_trick is 1:
            cardfirst = cardsTodistribute[0:currentPlayerHandsize]
            cardsecond = cardsTodistribute[(currentPlayerHandsize):(2*currentPlayerHandsize)]
            cardthird = cardsTodistribute[(2*currentPlayerHandsize):(3*currentPlayerHandsize-1)]

        if obs.nr_cards_in_trick is 2:
            cardfirst = cardsTodistribute[0:currentPlayerHandsize]
            cardsecond= cardsTodistribute[(currentPlayerHandsize):(2*currentPlayerHandsize-1)]
            cardthird= cardsTodistribute[(2*currentPlayerHandsize-1):(3*currentPlayerHandsize)]

        if obs.nr_cards_in_trick is 3:
            newarr = np.array_split(cardsTodistribute, 3)
            cardfirst= newarr[0]
            cardsecond= newarr[1]
            cardthird= newarr[2]
            
        # print('cardfirst: {} '.format(cardfirst))
        # print('cardsecond: {} '.format(cardsecond))
        # print('cardthird: {} '.format(cardthird))

        # print('playerfirst: {} '.format(playerfirst))
        # print('playersecond: {} '.format(playersecond))
        # print('playerthird: {} '.format(playerthird))

        for c in cardfirst:
            handsIn[playerfirst,c] = 1
        
        for c in cardsecond:
            handsIn[playersecond,c] = 1

        for c in cardthird:
            handsIn[playerthird,c] = 1

        
        # Before shuffle hands
        # print("before shuffle hand 1 {}".format(np.flatnonzero(handsIn[0])))
        # print("before shuffle hand 2 {}".format(np.flatnonzero(handsIn[1])))
        # print("before shuffle hand 3 {}".format(np.flatnonzero(handsIn[2])))
        # print("before shuffle hand 4 {}".format(np.flatnonzero(handsIn[3])))

        simulatedGamePre.init_from_cards(handsIn,obs.dealer)
        simulatedGamePre._state.player = currentPlayer
        simulatedGamePre._state.trump = obs.trump
        simulatedGamePre._state.forehand = obs.forehand
        simulatedGamePre._state.tricks = obs.tricks
        simulatedGamePre._state.trick_winner = obs.trick_winner
        simulatedGamePre._state.trick_points = obs.trick_points
        simulatedGamePre._state.trick_first_player = obs.trick_first_player
        simulatedGamePre._state.current_trick = obs.current_trick
        simulatedGamePre._state.nr_tricks = obs.nr_tricks
        simulatedGamePre._state.nr_cards_in_trick = obs.nr_cards_in_trick
        simulatedGamePre._state.nr_played_cards = obs.nr_played_cards
        simulatedGamePre._state.points[0] = obs.points[0]
        simulatedGamePre._state.points[1] = obs.points[1]

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
            # print('observed player {}'.format(playerNumber))
            simulatedGame=self.deterministRandomShuffle(simulatedGamePre)
            # print('Nr. of tricks in Game: {}'.format(simulatedGame._state.nr_tricks))
            # print('simulated Game Pre cards: {})'.format(simulatedGame._state.current_trick))
            # 
            # # Starting from the current game state we use the simulatedGame-Object to finish the Game
            # # the simulatedGame-Object returns the card to play which was able to end the game with the highest point Nr.
            
            # print("after shuffle hand 1 {}".format(np.flatnonzero(simulatedGame.state.hands[0])))
            # print("after shuffle hand 2 {}".format(np.flatnonzero(simulatedGame.state.hands[1])))
            # print("after shuffle hand 3 {}".format(np.flatnonzero(simulatedGame.state.hands[2])))
            # print("after shuffle hand 4 {}".format(np.flatnonzero(simulatedGame.state.hands[3])))
            
            finishedNode=monteCarloSimulation.findNextMove(simulatedGame, playerNumber) 
            winnerNodesList.append(finishedNode)
            # print('visits on a winner node: {}'.format(finishedNode.stat.getVisitCount()))
            
            # print('finished Game After cards: {})'.format(finishedGame._state.current_trick))

        # Here we determine the winner node with the most visits:
        bestNode=extractWinnerNode(winnerNodesList)
        finishedGame=bestNode.getState().getGame()
        # print('most visited node: {}'.format(bestNode.stat.getVisitCount()))

        # if the next move/card to play is still in the same trick, which means that, 
        # nr_tricks in the game remain unchanged when the next move is played.
         # then we play the last new entry in the trick of the finishedGame
        if simulatedGame._state.nr_tricks == finishedGame._state.nr_tricks :   
            # print('Same trick card: {}'.format(finishedGame._state.current_trick[finishedGame._state.nr_cards_in_trick -1]))
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
    

    