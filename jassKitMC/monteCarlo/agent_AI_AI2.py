# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import copy
import array
import math
import random
# https://www.journaldev.com/33185/python-add-to-array
from monteCarlo.montecarlo_incomplete import MonteCarloTreeSearchIncomplete
import numpy as np
from monteCarlo.node_montecarlo import Node
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, MAX_PLAYER, next_player, partner_player, color_of_card
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber
from jass.arena.arena import Arena
import tensorflow as tf
from tensorflow import keras
import traceback
import os 

# AI-modules
# Necessary to use the ML-model
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle



class AgentAIAI2 (Agent):
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
        print("Determining trump...")
        try:
            trump = self.action_trump_intern(obs)
            print("trump determined as '" + str(trump) + "'")
            return trump
        except Exception:
            traceback.print_exc()
            raise

    def action_trump_intern(self, obs: GameObservation) -> int:

        # trumpValues = {'DIAMONDS': 0, 'HEARTS': 1, 'SPADES': 2, 'CLUBS': 3, 'OBE_ABE': 4, 'UNE_UFE': 5}

        self._logger.info('Trump request')
        # The trained ML needs a 2D-array with categories and cards.
        # first we create the Data-Frame
        columns = card_strings
        columns = np.append(columns, 'FH')
        print("category of columns {}".format(columns))
        # Es gibt die Kategories und die Kartenwerte

        dataIn = copy.deepcopy(obs.hand)

        if obs.forehand == -1:
            dataIn = np.append(dataIn, 1)
        else:
            dataIn = np.append(dataIn, 0)

        # wir transponieren hier das dataIn
        dataIn = dataIn.reshape(1, 37)

        predictFrameX = pd.DataFrame(data=dataIn, index=[0], columns=columns, dtype=int)
        # The key values are in: card_strings
        print("predict Array {}".format(predictFrameX.head()))
        print("shape PredictX {}".format(predictFrameX.shape))

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # print("__file__ {}".format(__file__))
        # print("dir_path {}".format(dir_path))

        dir_path2= os.path.join(dir_path, 'model_ml')

        if obs.forehand == -1:
            print('directory path: {}'.format(dir_path))
            print('directory path: {}'.format(dir_path2))
            # if forehand is not yet set, we are the forehand player and can select trump or push
            # trained model with push-option True/False

            # loaded_model = pickle.load(open(os.path.join(dir_path, 'finalized_model.sav'), 'rb'))
            loaded_model = keras.models.load_model(dir_path2)

            # Let's check:
            # np.testing.assert_allclose(model.predict(test_input), reconstructed_model.predict(test_input))
            trump = loaded_model.predict(predictFrameX)
            # print('trump forehand is: {}'.format(np.argmax(trump[0])))

            # If the hot-encoded value is 6 then return 10 for push.
            if np.argmax(trump[0]) == 6:
                return 10

            # return trumpValues[trump[0]]
            return np.argmax(trump[0])
        # if not push or forehand, select a trump
        # This means: using a model, where the PUSH-option does not exist,
        # since this option really does not exist for the player to which the trump-decision is pushed.
        # Noch pushed und nicht-pushed unterscheiden und implementieren für ml-modell!
        dir_path2= os.path.join(dir_path, 'model_ml_back')
        loaded_model = keras.models.load_model(dir_path2)
        # loaded_model = pickle.load(open(os.path.join(dir_path, 'finalized_model_pushed.sav'), 'rb'))

        trump = loaded_model.predict(predictFrameX)
        print('trump backhand is: {}'.format(np.argmax(trump[0])))
        # return trumpValues[trump[0]]
        return int(np.argmax(trump[0]))

    # nextPlayerNr: The number of the player in the game (NORTH, SOUTH, WEST, EAST)
    # nextPlayerPosition: The position of the player in the tick

    def action_play_card(self, obs: GameObservation) -> int:
        print("Determining card to play...")
        try:
            card =  self.action_play_card_intern(obs)
            # print('hand of player: {} '.format(np.flatnonzero(obs.hand)))
            print("Card dermined as '" + str(card) + "'")
            return card
        except Exception:
            traceback.print_exc()
            raise

    def action_play_card_intern(self, obs: GameObservation) -> int:

        dir_path = os.path.dirname(os.path.realpath(__file__))
        # dir_path2= dir_path + '\modelcardplay_ml'
        # dir_path2= dir_path + '\modelcardplay_ml2'
        dir_path2= os.path.join(dir_path, 'modelcardplay_m2600')
        loaded_model = keras.models.load_model(dir_path2)
        # loaded_model = pickle.load(open(os.path.join(dir_path, 'finalized_model_pushed.sav'), 'rb'))

        # ********************* Here we create the Input data ******************************
        obsFeature = np.zeros([7, 36], np.int32)
        obsFeatureLong = np.array([])
        obsCurrentTrick = []
        obsCurrentTrickHot = np.zeros(36)
        obsPlayedCards = []
        obsPlayedCardsHot = np.zeros(36)
        obsPlayerHand = []
        obsPlayerHandValid = []
        obsTrumpValid = []
        obsTrumpPlayed = []
        obsTrumpTrick = []
        obsInput = []

        obsCurrentTrick = obs.current_trick
        # print('q {}'.format(q))   
        # print('current Trick {}'.format(obsCurrentTrick))       
        # print('trump {}'.format(c.trump)) 

        color_played = color_of_card[obs.current_trick[0]]
        # print('played color is: {}'.format(color_played))
        obsPlayerHand = obs.hand
        moveNr = obs.nr_cards_in_trick
        # print('player hand: {}'.format(obsPlayerHand))
        obsPlayerHandValidHot = RuleSchieber.get_valid_cards(RuleSchieber,obsPlayerHand, obsCurrentTrick, moveNr, obs.trump)
        # print('player hand hot valid: {}'.format(obsPlayerHand))
        obsPlayerHandValid = np.flatnonzero(obsPlayerHandValidHot)
        # print('player hand valid: {}'.format(obsPlayerHandValid))

        

        #Add the previously played cards
        obsPlayedCards = obs.tricks[obs.tricks>-1]
            
        for q in obsPlayedCards:
            if q>-1:
                obsPlayedCardsHot[q] = 1
                    
        for q in obsCurrentTrick:
            if q>-1:
                obsCurrentTrickHot[q] = 1
            
        # ******** The cards in the hand of the player **********
        for q in np.flatnonzero(obsPlayerHand):
            if q>-1:
                obsFeature[0,q] = 1
            
        # ******** The valid cards in the hand of the player ********
        for q in obsPlayerHandValid:
            if q>-1:
                obsFeature[1,q] = 1
                    
        # *********** We add the played Cards to the feature List *********
        for q in obsPlayedCards:
            if q>-1:
                obsFeature[2,q] = 1
        # print('Input features {}'.format(obsFeature))       
            
        # Current trick
        for q in obsCurrentTrick:
            if q>-1:
                obsFeature[3,q] = 1
        # print('Input features {}'.format(obsFeature))  
        # 
        # trumps among the valid cards in hand:
        print('trump {}'.format(obs.trump))  
        if obs.trump < 4:
            obsTrumpValid = np.flatnonzero(obsPlayerHandValidHot * color_masks[obs.trump, :])  
            obsTrumpPlayed = np.flatnonzero(obsPlayedCardsHot * color_masks[obs.trump, :])  
            obsTrumpTrick = np.flatnonzero(obsCurrentTrickHot * color_masks[obs.trump, :])  
            
        # ***************** Unten/Oben arrays **************************************************
        oben_mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int32)  
        oben_mask_hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int32)  
        
        highestD = -1
        highestH = -1
        highestS = -1
        highestC = -1
        lowestD = -1
        lowestH = -1
        lowestS = -1             
        lowestC = -1
     
        DNext =   [0,1,2,3,4,5,6,7,8]
        for q in range(0,9):
            if obsPlayedCardsHot[q] == 1:
                 DNext.remove(q)
                    
        HNext =   [9,10,11,12,13,14,15,16,17]
        for q in range(9,18):
            if obsPlayedCardsHot[q] == 1:
                HNext.remove(q)
            
        SNext =   [18, 19, 20, 21, 22, 23, 24, 25, 26]
        for q in range(18,27):
            if obsPlayedCardsHot[q] == 1:
                SNext.remove(q)
            
        CNext =   [27, 28, 29, 30, 31, 32, 33, 34, 35]
        for q in range(27,36):
            if obsPlayedCardsHot[q] == 1:
                    CNext.remove(q)
            
        if len(DNext)>0:
            # print('DNext{}'.format(DNext))
            highestD = DNext[np.argmin(DNext)]
            # print('Highest D{}'.format(highestD))
            oben_mask_hand[highestD] = 1
                
        if len(HNext)>0:
            # print('HNext{}'.format(HNext))
            highestH = HNext[np.argmin(HNext)]
            # print('Highest H{}'.format(highestH))
            oben_mask_hand[highestH] = 1
                
        if len(SNext)>0:
            # print('SNext{}'.format(SNext))
            highestS = SNext[np.argmin(SNext)]
            # print('Highest S{}'.format(highestS))
            oben_mask_hand[highestS] = 1
            
        if len(CNext)>0:
            # print('CNext{}'.format(CNext))
            highestC = CNext[np.argmin(CNext)]
            # print('Highest C{}'.format(highestC))
            oben_mask_hand[highestC] = 1     
                
        # print('DNext nach{}'.format(DNext))
        # print('HNext nach{}'.format(HNext))
        # print('SNext nach{}'.format(SNext))
        # print('CNext nach{}'.format(CNext))
                    
        # print('oben mask hand{}'.format(oben_mask_hand))
            
        # cards higher than the unplayed highest card
        if highestD > -1:
            for q in range(0, highestD):
                oben_mask[q] = 1

        if highestH > -1:    
            for q in range(9, highestH):
                oben_mask[q] = 1

        if highestS > -1:   
            for q in range(18, highestS):
                oben_mask[q] = 1

        if highestC > -1:   
            for q in range(27, highestC):
                oben_mask[q] = 1
                
        # print('oben mask{}'.format(oben_mask))
           
        if obs.trump == 4:
            obsTrumpValid = np.flatnonzero(obsPlayerHandValidHot * oben_mask_hand)  
            obsTrumpPlayed = np.flatnonzero(obsPlayedCardsHot * oben_mask)  
            obsTrumpTrick = np.flatnonzero(obsCurrentTrickHot * oben_mask)  
                
                
        unten_mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int32)  
        unten_mask_hand = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], np.int32)  

        if len(DNext)>0:
            # print('DNext{}'.format(DNext))
            lowestD = DNext[np.argmax(DNext)]
             # print('Highest D{}'.format(highestD))
            unten_mask_hand[lowestD] = 1
                
        if len(HNext)>0:
            # print('HNext{}'.format(HNext))
            lowestH = HNext[np.argmax(HNext)]
            # print('Highest H{}'.format(highestH))
            unten_mask_hand[lowestH] = 1
                
        if len(SNext)>0:
            # print('SNext{}'.format(SNext))
            lowestS = SNext[np.argmax(SNext)]
            # print('Highest S{}'.format(highestS))
            unten_mask_hand[lowestS] = 1
            
        if len(CNext)>0:
            # print('CNext{}'.format(CNext))
            lowestC = CNext[np.argmax(CNext)]
            # print('Highest C{}'.format(highestC))
            unten_mask_hand[lowestC] = 1     
                
        # print('DNext nach{}'.format(DNext))
        # print('HNext nach{}'.format(HNext))
        # print('SNext nach{}'.format(SNext))
        # print('CNext nach{}'.format(CNext))
                    
        # print('oben mask hand{}'.format(oben_mask_hand))
            
        # cards lower than the unplayed lowest card
        if lowestD > -1:
            for q in range(lowestD+1,9):
                unten_mask[q] = 1

        if lowestH > -1:     
            for q in range(lowestH+1,18):
                unten_mask[q] = 1

        if lowestS > -1:       
            for q in range(lowestS+1,27):
                unten_mask[q] = 1

        if lowestC > -1:       
            for q in range(lowestC+1,36):
                unten_mask[q] = 1    
                
            
                
        unten_mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], np.int32)                     
        if obs.trump == 5:
            for s in range(0,36):
                obsTrumpValid = np.flatnonzero(obsPlayerHandValidHot * unten_mask_hand)  
                obsTrumpPlayed = np.flatnonzero(obsPlayedCardsHot * unten_mask)  
                obsTrumpTrick = np.flatnonzero(obsCurrentTrickHot * unten_mask)    
        # ***************** ende Unten/Oben arrays **************************************************


        # We add the trumps among the handcards to the list
        for q in obsTrumpValid:
            if q>-1:
                obsFeature[4,q] = 1
                # print('trump added {}'.format(obsFeature[q]))
         
        # *********** We add the trumps among the played Cards to the feature List *********
        for q in obsTrumpPlayed:
            if q>-1:
                obsFeature[5,q] = 1
        # print('Input features {}'.format(obsFeature))   
    
        # We add the trumps among the current trick cards to the list
        for q in obsTrumpTrick:
            if q>-1:
                obsFeature[6,q] = 1  

        # ********************* The created Input Data *************************************
        for q in range(7):
            obsFeatureLong = np.concatenate((obsFeatureLong,np.array(obsFeature[q], dtype=np.int32)))

        columns = card_strings
        for q in range(6):
            columns = np.concatenate((columns,card_strings))

        # print('feature long 0 {}'.format(np.flatnonzero(obsFeatureLong)))
        # obsInput.append(obsFeatureLong) 
        obsInput = obsFeatureLong
        obsInput = obsInput.reshape(1, 7*36)

        predictFrameX = pd.DataFrame(data=obsInput, index=[0], columns = columns, dtype=int)
        # print('obsInput: {}'.format(obsInput))
        cardtoPlay = loaded_model.predict(obsInput)[0]
        # print('all cards to play: {}'.format(cardtoPlay))
        # print('card to Play: {}'.format(np.argmax(cardtoPlay)))
        # print('card to Play2: {}'.format(np.argmax(cardtoPlay)))

        for q in range(0, len(cardtoPlay)):
            # print('q: {}'.format(q))
            if q not in obsPlayerHandValid:
                # print('q invalid: {}'.format(q))
                cardtoPlay[q] = 0

        # print('handcard filtered: {}'.format(cardtoPlay))
        # Valid cards to play among all cards
        validCardtoPlay = RuleSchieber.get_valid_cards(RuleSchieber,cardtoPlay, obsCurrentTrick, moveNr, obs.trump)
        # Here we have to make sure, that the cards are in the hand of the player:
        # print('validCardtoPlay: {}'.format(validCardtoPlay))
        # print('handvalid before: {}'.format(obsPlayerHandValid))
        # print('len valid: {}'.format(len(validCardtoPlay)))
        # Out of the cards in the hand of the player, we have to choose the best one:
        # print('all valid cards: {}'.format(validCardtoPlay))
        # print('valid card to Play: {}'.format(np.argmax(validCardtoPlay)))        

        # return trumpValues[trump[0]]
        return np.argmax(validCardtoPlay)


    

    