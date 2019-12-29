#Training the bot using stories.md
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import logging
import asyncio 
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core import config as policy_config

if __name__ == '__main__':
	logging.basicConfig(level='INFO')
	
	training_data_file = './data/stories.md'
	model_path = './models/dialogue'
	#policies2 = policy_config.load("policies.yml")
	agent = Agent("weather_domain.yml",policies=[MemoizationPolicy(), KerasPolicy(epochs=200,batch_size=10,validation_split=0.2)])
	training_data = agent.load_data(training_data_file)
	agent.train(training_data)
	agent.persist(model_path)
