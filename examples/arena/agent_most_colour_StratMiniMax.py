# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, MAX_PLAYER, next_player, partner_player
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.rule_schieber import RuleSchieber
import copy


class AgentMostColourMinMax (Agent):
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


    # We cheat and access as Player the total GameState
    def setStateObject(self, stat: GameState):
        self.state = stat

        # Action_play_card uses MiniMax-Strategy for each tick.
    # We need additionally stat, since we have to know the cards of 
    # the subsequent players to create a minimax tree
    # self: the Player
    # playedCard: The played card
    # playerNr: the number of the child/next player
    # depth: number of remaining ticks
    # maximizingPlayer: true or false
    # return the value for the team
    def minimax (self, observedPlayer: int, nextPlayerNr: int, nextPlayerPosition: int,trickArray: np.array,depth: int, maximizingPlayer: bool):
        valid_cards = np.flatnonzero(self._rule.get_valid_cards(self.state.hands[nextPlayerNr,:],trickArray,nextPlayerPosition,self.state.trump))
        # print('hands {} '.format(np.flatnonzero(self.state.hands)))
        # print('num {} is depth'.format(depth)) #return bestValue
        
        if depth == 0:
            # After the last card has been played, we evaluate the full trickArray

            # print('{} final fill'.format(trickArray)) #return bestValue

            # Here we calculate the value of the tick for the team:
            # The tricks which have been already played.
            if self.state.nr_tricks == 8:
                # If it is the last trick
                is_last = True
            else: is_last = False

            # From rule_schieber we can use:
            points = self._rule.calc_points(trickArray, is_last, self.state.trump)
            winner = self._rule.calc_winner(trickArray, self.state.trick_first_player[self.state.nr_tricks], self.state.trump)
            # And calculate so the hypotetical outcome when we would play through the tree up to the leafe.

            # print('points {} '.format(points)) #return bestValue
            # A leaf does not get a cardnumber, but a -1, since it only contains the value analysis
            if winner == observedPlayer:
                # print('{} observed player'.format(observedPlayer))
                return [points,-1]
                
            if winner == partner_player[observedPlayer]:
                # print('{} partner leaf'.format(partner_player[observedPlayer]))
                return [points,-1]

            # If the team of the observed Player loses in this tick, then there are no points in the leafe.
            return [0,-1]
          
            # ****************************************************
            # print('finished') #return bestValue
        
        if maximizingPlayer:
            bestValue = [-10**10,0] # is -10^10
            for c in valid_cards:
                trickArray[nextPlayerPosition]=c
                # print('num {} depth maximizer'.format(trickArray)) #return bestValue
                value = [self.minimax(observedPlayer,next_player[nextPlayerNr],nextPlayerPosition+1,trickArray,depth-1,False)[0],c]
                # print('{} value maximizer'.format(value))
                # print('{} BestValue maximizer'.format(bestValue))
                bestValue = [max(bestValue[0],value[0]),c]
                # print('{} new BestValue maximizer'.format(bestValue))
            return bestValue
                
        else :
            bestValue = [10**10,0] # is 10^10
            for c in valid_cards:
                trickArray[nextPlayerPosition]=c
                # print('num {} depth minimizer'.format(trickArray)) #return bestValue
                value = [self.minimax(observedPlayer,next_player[nextPlayerNr],nextPlayerPosition+1,trickArray,depth-1,True)[0],c]
                # print('{} value minimizer'.format(value))
                # print('{} BestValue minimizer'.format(bestValue))
                bestValue = [min(bestValue[0],value[0]),c]
                # print('{} new BestValue minimizer'.format(bestValue))
            return bestValue

    def action_play_card(self, obs: GameObservation) -> int:
        depth = (MAX_PLAYER+1) - obs.nr_cards_in_trick
        # For the evaluation we need to know which of the cards in the
        # trick belongs to the player. If he is the first player then
        # his card in the tickArray is 0.
        playerPosition = obs.nr_cards_in_trick
        playerNumber = obs.player
        # an array of four values with the already played cards.
        # this array is needed in the minimax to evaluate finally the points.
        # the trickarray corresponds to the node.
        trickArray=obs.current_trick

        trickArray=copy.deepcopy(obs.current_trick)
        firstPlayer=self.state.trick_first_player[self.state.nr_tricks] 
        trumpOfGame=self.state.trump
        handsOfGame = copy.deepcopy(self.state.hands)
        valid_cards = np.flatnonzero(self._rule.get_valid_cards(handsOfGame[playerNumber,:],trickArray,playerPosition,trumpOfGame))
        all_cards = np.flatnonzero(handsOfGame[playerNumber,:])
        #print('{} player position before'.format(playerPosition))
        # print('{} trumpOfGame before'.format(trumpOfGame))
        # print('{} trick array before'.format(trickArray))
        # print('{} all cards before'.format(all_cards))
        # print('{} valid cards before'.format(valid_cards))





        card = self.minimax(obs.player,playerNumber,playerPosition,trickArray,depth,True)[1]


        valid_cards = np.flatnonzero(self._rule.get_valid_cards(handsOfGame[playerNumber,:],trickArray,playerPosition,trumpOfGame))
        all_cards = np.flatnonzero(handsOfGame[playerNumber,:])
        # print('{} player position after'.format(playerPosition))
        # print('{} trumpOfGame after'.format(trumpOfGame))
        # print('{} trick array after'.format(obs.current_trick))
        # print('{} all cards after'.format(all_cards))
        # print('{} valid cards after'.format(valid_cards))
        # print('{} returned card'.format(card))


        return card




    