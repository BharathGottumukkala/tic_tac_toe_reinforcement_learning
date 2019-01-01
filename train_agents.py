import gym, gym_tic_tac_toe
import time
import random
from pprint import pprint
import pickle
import numpy as np
from copy import deepcopy

class Agent():
	def __init__(self, name):

		self.Q = {} # Q Table
		self.__gamma = 0.9 # discount factor
		self.__alpha = 0.9 # learning rate
		self.__exploration_prob = 0.9
		self.__exploration_decay = 0.99999

		self.trainable = True

	def play_move(self, state):

		if state not in self.Q:
			# Create a new state key if it is not present in our dictionary
			self.Q[state] = np.array([0.0] * 9)
		# Select the optimal action / move for a given state 
		action = self.max_Q_value(state)

		return action

	def update_Q_table(self, state, action, reward, next_state):
		if self.trainable:
			if state not in self.Q:
				# Create a new state key if it is not present in our dictionary
				self.Q[state] = np.array([0.0] * 9)
			if next_state not in self.Q:
				""" 
					We can initialize the Q_table entry of the next state
					as the current state and then update it in the further steps
				"""
				self.Q[next_state] = deepcopy(self.Q[state])

			# Bellman's Equation to update the Q-Table
			self.Q[state][action] += self.__alpha * (reward + self.__gamma*np.max(self.Q[next_state]) - self.Q[state][action])

	def max_Q_value(self, state):
		"""
		There is a trade off between the agent trying to explore the environment
		and the agent trying to exploit. 

		This trade off is balanced here by using a decay in the exploration factor
		which will let the agent exploit after approximately 50% of its training 
		"""
		# Decrease the exploring probability by a factor
		self.__exploration_prob *= self.__exploration_decay
		if random.random() < self.__exploration_prob:
			# Pick a random action to explore
			pick = random.randint(0, 8)
			return pick

		else:
			# Pick the action withmaximum value in the table whie exploiting
			return self.Q[state].argmax()

	def act_random(self):
		self.__exploration_prob = 1
		self.__exploration_decay = 1

	def get_better(self, explore_prob=0.9, explore_decay=0.99999):
		self.__exploration_prob = explore_prob
		self.__exploration_decay = explore_decay

	def exploit(self):
		self.__exploration_prob = 0

	def print_Q_table(self):
		pprint(self.Q)
		print()

	def restore_Q(self, path):
		# Load an already trained models Q_Table
		rest_from = open(path, 'rb')
		self.Q = pickle.load(rest_from)
		rest_from.close()

	def dump_Q(self, path):
		# Dump the Q_Table after training
		dump_loc = open(path, 'wb')
		pickle.dump(self.Q, dump_loc)
		dump_loc.close()

env = gym.make("tic_tac_toe-v0")
agent1 = Agent('Neo') #for X

agent2 = Agent('Trinity') #for O



# [victories, draws]
# agent1_counts = [0, 0]
# agent2_counts = [0, 0]

def train_agents():
	# We train the first player against a completely random opponent
	# Now we make the agent 2 pick completely randomm moves by making the 
	# exploration probability 1.
	# agent1_counts = [0, 0]
	# agent2_counts = [0, 0] 
	# global agent1_counts, agent2_counts
	#for restoration
	restore = True
	# To dump
	dump = True
	if restore:
		try:
			agent1.restore_Q(path='Neo.pkl')
			agent2.restore_Q(path='Trinity.pkl')
			
		except Exception as e:
			print("Couldnt restore Q table", e)

	for agent in range(1,3):
		if agent == 1:
			agent2.act_random()
			agent1.get_better()

		elif agent == 2:
			agent1.act_random()
			agent2.get_better()

		start_time = time.time()
		episodes = 400000

		inform = episodes//100

		#training
		for episode in range(episodes):
			if episode%inform == 0:
				print("Episode number -> {}/{}".format(episode, episodes))

			run_episode()

		
		print("Finished training agent {} in {}".format(agent, time.time() - start_time))
	#dump the Q table
	if dump:
		agent1.dump_Q(path='Neo.pkl')
		agent2.dump_Q(path='Trinity.pkl')

	
def run_episode():
	state = env.reset()
	done = False
	turn = {'turn': 'X'}
	prev_state = state
	prev_action = -1
	while not done:
		############
		# Player 1 #
		############
		if turn['turn'] == 'X':
			action = agent1.play_move(state)
			next_state, reward, done, turn = env.step(action)

			if reward == -100: # Illegal move update
				agent1.update_Q_table(state, action, reward, next_state)

			elif reward == 1: # Win condition update
				agent1.update_Q_table(state, action, reward, next_state)
				agent2.update_Q_table(prev_state, prev_action, -reward, state)

			else: # Propagation Update
				if prev_state != state:
					agent2.update_Q_table(prev_state, prev_action, reward, next_state)

		############
		# Player 2 #
		############
		elif turn['turn'] == 'O':
			action = agent2.play_move(state)
			next_state, reward, done, turn = env.step(action)

			if reward == -100: # Illegal move update
				agent2.update_Q_table(state, action, reward, next_state)
			elif reward == 1: # Win condition update
			# else:
				agent2.update_Q_table(state, action, reward, next_state)
				agent1.update_Q_table(prev_state, prev_action, -reward, state)
			else: # Propagation Update
				if prev_state != state:
					agent1.update_Q_table(prev_state, prev_action, reward, next_state)

		prev_action = action
		prev_state = deepcopy(state)
		state = next_state					
	
if __name__ == '__main__':	
	train_agents()
