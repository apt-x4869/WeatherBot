from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import asyncio

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.policies import FallbackPolicy
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.training import interactive
from rasa.utils.endpoints import EndpointConfig
from rasa_core import config as policy_config


logger = logging.getLogger(__name__)


def run_weather_online(interpreter,
                          domain_file="weather_domain.yml",
                          training_data_file='data/stories.md'):
    policies2 = policy_config.load("config.yml")
    action_endpoint = "endpoint.yml"					  
    agent = Agent(domain_file,policies=policies2,interpreter=interpreter,action_endpoint=action_endpoint)
    				  
    data = asyncio.run(agent.load_data(training_data_file))			   
    agent.train(data)
    interactive.run_interactive_learning(agent,training_data_file)
    return agent


if __name__ == '__main__':
    logging.basicConfig(level="DEBUG")
    nlu_interpreter = NaturalLanguageInterpreter.create('./models/nlu/weathernlu')
    run_weather_online(nlu_interpreter)
