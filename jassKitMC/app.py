# HSLU
#
# Created by Thomas Koller on 10.09.20
#
import logging

from monteCarlo.agent_AI_Monte_Carlo_incomplete import AgentMonteCarloAIIncomplete
from monteCarlo.agent_AI_AI import AgentAIAI 
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.service.player_service_app import PlayerServiceApp


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('player_service')

    # you could use a configuration file to load additional variables
    # app.config.from_pyfile('my_player_service.cfg', silent=False)

    # add some players
    app.add_player('random', AgentRandomSchieber())
    app.add_player('monte-carlo', AgentMonteCarloAIIncomplete())
    app.add_player('ai-player',AgentAIAI())

    return app


app = create_app()
