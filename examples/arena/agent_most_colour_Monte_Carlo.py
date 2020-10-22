# HSLU
#
# Created by Thomas Koller on 7/28/2020
#
import logging
import copy
from montecarlo import MonteCarloTreeSearch
import numpy as np
from jass.agents.agent import Agent
from jass.game.const import PUSH, MAX_TRUMP, card_strings, color_masks, MAX_PLAYER, next_player, partner_player
from jass.game.game_observation import GameObservation
from jass.game.game_state import GameState
from jass.game.game_sim import GameSim
from jass.game.rule_schieber import RuleSchieber


class AgentMostColourMonteCarlo (Agent):
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

    # nextPlayerNr: The number of the player in the game (NORTH, SOUTH, WEST, EAST)
    # nextPlayerPosition: The position of the player in the tick

    def action_play_card(self, obs: GameObservation) -> int:

        monteCarloSimulation = MonteCarloTreeSearch()
        # We prepare the mock-game for the Monte-Carlo-Simulation
        simulatedGame = GameSim(self._rule)
        # We initialize the Game by the state
        simulatedGame.init_from_state(self.state)

        # When the action_play_card we play a card. Now we determine,
        # which player we are (North, South, West, East)
        playerNumber = obs.player
        print('Nr. of tricks in Game: {}'.format(simulatedGame._state.nr_tricks))
        # print('simulated Game Pre cards: {})'.format(simulatedGame._state.current_trick))

         # Starting from the current game state we use the simulatedGame-Object to finish the Game
        # the simulatedGame-Object returns the card to play which was able to end the game with the highest point Nr.
        finishedGame=monteCarloSimulation.findNextMove(simulatedGame, playerNumber) 


        # print('finished Game After cards: {})'.format(finishedGame._state.current_trick))
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

    