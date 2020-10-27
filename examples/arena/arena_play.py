# HSLU
#
# Created by Thomas Koller on 7/28/2020
#

import logging
import sys
import numpy as np
import array
from jass.agents.monteCarlo.node_montecarlo import Node
# Dadurch werden im Programmrun nicht die dateien im
from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.monteCarlo.agent_AI_Monte_Carlo_incomplete import AgentMonteCarloAIIncomplete


def main():

    # Set the global logging level (Set to debug or info to see more messages)
    logging.basicConfig(level=logging.WARNING)

    # setup the arena
    arena = Arena(nr_games_to_play=100, save_filename='arena_games')

    player = AgentRandomSchieber()
    my_player = AgentMonteCarloAIIncomplete()

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
