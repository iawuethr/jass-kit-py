# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, card_values, color_of_card
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class AgentMostColourStratDeal1 (Agent):
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
        """
        Select trump randomly. Pushing is selected with probability 0.5 if possible.
        Args:
            obs: the current match
        Returns:
            trump action
        """
        self._logger.info('Trump request')
        if obs.forehand == -1:
            # if forehand is not yet set, we are the forehand player and can select trump or push
            if self._rng.choice([True, False]):
                self._logger.info('Result: {}'.format(PUSH))
                return PUSH
        # if not push or forehand, select a trump
        trump = 0
        # print('Number of cards {} '.format(len(obs.hand)))
        max_number_in_color = 0
        for c in range(4):
            number_in_color = (obs.hand * color_masks[c]).sum()
            if number_in_color > max_number_in_color:
                max_number_in_color = number_in_color
                trump = c
        return trump

    def action_play_card(self, obs: GameObservation) -> int:
        """
        Select randomly a card from the valid cards
        Args:
            obs: The observation of the jass match for the current player
        Returns:
            card to play
        """
        self._logger.info('Card request')
        # cards are one hot encoded
        valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # convert to list and draw a value
        # decoded cards
        decodedcards = np.flatnonzero(valid_cards)
        # contains a 1 at the index where a valid card is positioned
        # print('num {} valid cards'.format(valid_cards))
        # contains the index of the valid cards
        # print('num {} decoded cards'.format(decodedcards))
        # contains the values of all cards in the game and hands for the current tick
        # in a 4x9 matrix
        # print('num {} card_values'.format(card_values[obs.declared_trump]))
        decodedcardsValues = valid_cards * card_values[obs.declared_trump]
        # print('num {} decodedcardsValues of valid cards'.format(decodedcardsValues))
        # valid non-trump cards (if there is a color-trump)
        # if a trump is played as first card then colorcards = trumpcards
        color_played = color_of_card[obs.current_trick[0]]
        # The colorcards in the 4x9, where a 1 is set when there is a card with the colour of the trick
        colorcards = valid_cards * color_masks[color_played]
        # The indices of the trumpcards
        trumpcards = np.zeros(shape=36, dtype=np.int32)

        if obs.declared_trump<5:
            # valid trump-cards 4x9
            trumpcards = valid_cards * color_masks[obs.declared_trump]
            # the values of the trump cards at the given indices 4x9
            trumpcardsValue= trumpcards * card_values[obs.declared_trump]

        # If the value of a colorcard is bigger than the played cards, play this colorcard
        # a 4x9-vector with the values of the cards at the indices where there is a colour of the tick.
        colorcardsvalue = (colorcards * card_values[color_played])
        
        # get the cards of the trick on the playground, which have the color of the trick.
        # The 4-array contains the indices of the cards
        # print('num {} current trick'.format(obs.current_trick))

        self.currentTrickEncoded = np.zeros(shape=36, dtype=np.int32)
        # Encode the current trick in a 4x9 matrix
        for c in range(len(obs.current_trick)):
            if obs.current_trick[c] >= 0:
                # print('num {} TrickCard'.format(obs.current_trick[c]))
                self.currentTrickEncoded[obs.current_trick[c]]=1

        # print('num {} currentTrickEncoded'.format(self.currentTrickEncoded))
        # print('num {} colorofTick'.format(color_played))
        # the cards where the colour is actually played in the trick laying on the playground ("aagaeh")
        colorCardsTrick = self.currentTrickEncoded * color_masks[color_played]
        # print('num {} played colorcards of trick'.format(colorCardsTrick))
        # get the largest value out of these colorcards laying on the playground
        colorCardsTrickValue = colorCardsTrick * card_values[color_played]
        # print('num {} value of colorcards laying on playground'.format(colorCardsTrickValue))
        maxValue=np.max(colorCardsTrickValue)

        # play the first colourcard, whose value is above the given value. 
        # rule can be improved by taking into account that the card 10 has a high value but a low strength.
        for c in range(len(colorcardsvalue)):
            if colorcardsvalue[c] > maxValue:
                self._logger.info('Played card: {}'.format(card_strings[c]))
                # the return value must be decoded
                # print('num {} colorPlayed'.format(c))
                # returns index of the card
                return c

        # otherwise use a trump if you have a trump, since obviously there are
        # very strong and valuable cards in the play.
        # use the trumpcard with the lowest value.

        # check if there is at all a trumpcard. If so, use it.
        # use the one with the smallest value.

        # Trumpcards values are 4x9 and there are Zeros in cases where there is no Trump
        # --> this leads to the fact, that Card 35 is always the trump with the min-value!!!
        # use decoded value range instead, like in Other cards!!!!
        if sum(trumpcardsValue) > 0:
            # print('num {} trump cards'.format(trumpcards))  
            # print('num {} trump cards values'.format(trumpcardsValue))  
            self.indexMinTrumpValue = 0
            self.minTrumpValue = 100
            trumpcardsDecoded = np.flatnonzero(trumpcards)
            for c in range(len(trumpcardsDecoded)):
                if trumpcardsValue[trumpcardsDecoded[c]] < self.minTrumpValue:
                    self.minTrumValue = trumpcardsValue[trumpcardsDecoded[c]] 
                    self.indexMinTrumpValue = trumpcardsDecoded[c]
            # print('num {} TrumpPlayed'.format(self.indexMinTrumpValue))    
            # returns index of the card 
            return self.indexMinTrumpValue

        # else, if there is no trump we can play any valid card.
        # we use the card with the lowest value  amongst the valid cards.
        # decodedcards: an array with the indices of the valid cards
        # print('num {} decodedcardsValues'.format(decodedcardsValues))    
        self.indexMinValidCard = 0
        self.minValidCardValue = 100
        for c in range(len(decodedcards)):
            if decodedcardsValues[decodedcards[c]] < self.minValidCardValue:
                self.minValidCardValue = decodedcardsValues[decodedcards[c]]
                self.indexMinValidCard = decodedcards[c]
        # print('num {} OtherCardPlayed'.format(self.indexMinValidCard))    
        # returns index of the card 
        return self.indexMinValidCard

