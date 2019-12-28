from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

from rasa_core import train,utils
#from rasa_core.training import online
from rasa_core.agent import Agent
from rasa_core.interpreter import NaturalLanguageInterpreter

from rasa_core.channels.console import CmdlineInput
from rasa_core.interpreter import RegexInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core import config as policy_config
from rasa_core.training import interactive

policies = policy_config.load("policies.yml")

logger = logging.getLogger(__name__)

'''
def run_weather_online(input_channel, interpreter,
                          domain_file="weather_domain.yml",
                          training_data_file='data/stories.md'):
    agent = Agent(domain_file,policies=policies)

    agent.train(training_data_file,
                       input_channel=input_channel,
                       max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)

    return agent
'''

def run_weather_online(interpreter):
    return train.train_dialog_model(
        domain_file="weather_domain.yml",
        stories_file='data/stories.md',
        output_path="models/dialog",
        endpoints="endpoint.yml",
        max_history=2,
        kwargs={
            "batch_size":50,
            "epochs":200,
            "max_training_samples":300
        }
    )

if __name__ == '__main__':
    logging.basicConfig(level="INFO")
    #utils.configure_colored_logging(loglevel="DEBUG")
    nlu_interpreter = NaturalLanguageInterpreter.create('./models/nlu/weathernlu')
    agent = run_weather_online(nlu_interpreter)
    online.serve_agent(agent)