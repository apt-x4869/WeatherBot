from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import asyncio

import logging

from rasa_core.agent import Agent
from rasa_core.channels.console import CmdlineInput
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core import config as policy_config
from rasa_core.training import interactive
from rasa_core.utils import EndpointConfig


logger = logging.getLogger(__name__)


def run_weather_online(input_channel, interpreter,domain_file="weather_domain.yml",training_data_file='data/stories.md'):

    #policies2 = policy_config.load("config.yml")
    action_endpoints = EndpointConfig(url="http://localhost:5055/webhook")
    agent = Agent("weather_domain.yml",interpreter=interpreter, policies=[MemoizationPolicy(), KerasPolicy(epochs=200,batch_size=50)],action_endpoint=action_endpoints)
    #data = asyncio.run(agent.load_data(training_data_file))
    data = agent.load_data(training_data_file)
    agent.train(data)
    interactive.run_interactive_learning(agent,training_data_file)
    return agent

if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    nlu_interpreter = NaturalLanguageInterpreter.create('./models/nlu/default/weathernlu')
    run_weather_online(CmdlineInput(), nlu_interpreter)
