import gym, gym_tic_tac_toe
import time
import random
from pprint import pprint
import pickle
import numpy as np
from copy import deepcopy
from train_agents import Agent

def make_move(Champ, human_player, env):

	Champ.exploit()
	state = env.reset()
	if human_player == 'X':
		env.render()
	prev_state = state
	done = False
	turn = {'turn': 'X'}
	while not done:
		if turn['turn'] == human_player:
			print("Select your move from {} - {}".format(0, 8))
			action = int(input())
			next_state, reward, done, turn = env.step(action)
			if reward == 1:
				print(Champ.name+ " -> You have won. Well done...")

		else:
			print(Champ.name + " is making its move")
			time.sleep(1)
			action = Champ.play_move(state)
			next_state, reward, done, turn = env.step(action)
			if reward == 1:
				print(Champ.name+ " -> Better luck next time. Though luck wont let you beat me. HAHAHAH")
		# print(turn)
		state = next_state
		env.render()
	if reward == 0:
		print(Champ.name+ " -> Is this the best you can do?")


def play():
	while True:
		print("Choose Player : ")
		print("1. X		2. O ")
		print("X will be playing first")

		human_player = int(input())
		if human_player == 1:
			human_player = 'X'
		elif human_player == 2:
			human_player = 'O'
		else:
			print("Please select one of (X , O)")
			continue

		if human_player == 'X':
			bot = 'O'
		else:
			bot = 'X'
		start_player = 'X'
		env = gym.make("tic_tac_toe-v0")
		Champ = Agent('Champ')
		if bot == 'O':
			Champ.name = "Trinity"
			Champ.restore_Q('Trinity.pkl')
		else:
			Champ.name = 'Neo'
			Champ.restore_Q('Neo.pkl')

		make_move(Champ, human_player, env)
				

play()