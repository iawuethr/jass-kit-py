# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import sys
import numpy as np
import copy
import array
from node_montecarlo import Node
# Dadurch werden im Programmrun nicht die dateien im 
sys.path.insert(0,"C:/Studium_HSLU/HS_2020/DLFORGAMES/JASSKIT/JASSKIT")
from agent_most_colour import AgentMostColour
from agent_highest_value import AgentHighestValue
from agent_highest_value_StratDeal1 import AgentHighestValueStratDeal1
from agent_most_colour_StratDeal1 import AgentMostColourStratDeal1
from agent_most_colour_StratMiniMax import AgentMostColourMinMax
from agent_most_colour_StratMiniMax2 import AgentMostColourMinMax2
from agent_most_colour_Monte_Carlo import AgentMostColourMonteCarlo
from agent_most_colour_Monte_Carlo_incomplete import AgentMostColourMonteCarloIncomplete
from agent_AI_Monte_Carlo_incomplete import AgentMonteCarloAIIncomplete 
from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.game.const import color_masks, card_strings
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_sim import GameSim
from jass.game.game_sim import GameState


def main():

    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=1, save_filename='arena_games')
    # player = AgentRandomSchieber()
    # player = AgentMostColour()
    # The Object of the GameState
    stateObject=arena._game.state

    # player = AgentMostColourStratDeal1()
    player = AgentRandomSchieber()
    # player = AgentRandomSchieber()
    # player = AgentMostColour()
    # my_player = AgentHighestValueStratDeal1()
    # my_player = AgentMostColour()
    # my_player = MyAgent()
    # my_player = AgentMostColourMinMax2()
    # my_player = AgentMostColourMonteCarlo()
    my_player = AgentMonteCarloAIIncomplete()

        # We provide the player with the gamestate
    my_player.setStateObject(stateObject)

    # arena.set_players(my_player, player, my_player, player)
    # game 1: The first player is the dealer, the second chooses the trump
    arena.set_players(player, my_player, player, my_player)
    print('Playing {} games'.format(arena.nr_games_to_play))
    arena.play_all_games()
    print('Average Points Team 0: {:.2f})'.format(arena.points_team_0.mean()))
    print('Average Points Team 1: {:.2f})'.format(arena.points_team_1.mean()))
    
    # Print system path to modules
    # for d in sys.path:
    #     print(d)


if __name__ == '__main__':
    main()
