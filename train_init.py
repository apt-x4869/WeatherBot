#Training the bot using stories.md
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging
import asyncio
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy

if __name__ == '__main__':
	logging.basicConfig(level='INFO')
	
	training_data_file = './data/stories.md'
	model_path = './models/dialogue'
	
	agent = Agent('weather_domain.yml', policies = [MemoizationPolicy(), KerasPolicy()])
	data = asyncio.run(agent.load_data(training_data_file))
	agent.train(data)
			
	agent.persist(model_path)