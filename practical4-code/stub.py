# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epsilon = 1 # Wasay: SARSA and Qlearn parameter
        self.gamma = 0.5 # Wasay: discount factor
        self.alpha = 0.2 # Wasay: learning rate
        self.bin_width = 100 # Wasay: set the bin width
        self.Q = {} # Wasay: the Q-value dictionary Q[state] = [r_a_1,r_a_2]

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        print "restarted"

    # Wasay: The function that bins the states:

    def bin_state(self,state):
        # Wasay: The binning procedure is as follows:

        ## 1: Take all the numbers from the state put it in an array and divide the array by 
        ##### bin_width.

        binned_state = np.array(state['monkey'].values()+state['tree'].values()) / self.bin_width  
       
        ## 2: Convert this array into a string

        str_binned_state = np.array_str(binned_state)

        return str_binned_state
    
    # Wasay: SARSA

    def SARSA(self,b_last_state,last_action,reward,b_current_state):
        
        # Epsilon greedy action:
        if npr.rand()<self.epsilon:
            # Pick a greedy action
            current_action = np.argmax(self.Q[b_current_state])
        else:
            # Pick a random action
            current_action = npr.rand()<0.5

        # Update Q

        Q_ls_la = self.Q[b_last_state][last_action] # Q(S,a)
        Q_cs_ca = self.Q[b_current_state][current_action] # Q(S',a')

        self.Q[b_last_state][last_action]= Q_ls_la + self.alpha*((self.last_reward + self.gamma*(Q_cs_ca)) - Q_ls_la)

        return current_action

    # Wasay: Q-learning

    def Qlearn(self,b_last_state,last_action,reward,b_current_state):


        # Update Q

        Q_ls_la = self.Q[b_last_state][last_action] # Q(S,a)
        max_Q_cs_ca = max(self.Q[b_current_state]) # max_a Q(S',a')

        self.Q[b_last_state][last_action]= Q_ls_la + self.alpha*((self.last_reward + self.gamma*(max_Q_cs_ca)) - Q_ls_la)

        # Epsilon greedy action:
        if npr.rand()<self.epsilon:
            # Pick a greedy action
            current_action = np.argmax(self.Q[b_current_state])
        else:
            # Pick a random action
            current_action = npr.rand()<0.5

        return current_action



    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        # If the first observed state, make appropriate changes and return 
        ## a random action:

        if self.last_state==None:
            current_action = npr.rand() < 0.1
            current_state  = state

            b_current_state = self.bin_state(state)
            self.Q[b_current_state]=[npr.rand(),npr.rand()]

            self.last_action = current_action
            self.last_state  = current_state

            return self.last_action

        # Otherwise

        b_current_state = self.bin_state(state)
        b_last_state = self.bin_state(self.last_state)
        
        # Wasay: Check if we are encountering this state for the first time or we have seen 
        ## it before:

        if b_current_state not in self.Q.keys():
            # Wasay: Initialize arbitrarily
            #self.Q[b_current_state]=[npr.rand(),npr.rand()]
            self.Q[b_current_state]=[0,0]
        else: 
            print "state repeated"


        
        # You might do some learning here based on the current state and the last state.

        current_action = self.SARSA(b_last_state,self.last_action,self.last_reward,b_current_state)
        #current_action = self.Qlearn(b_last_state,self.last_action,self.last_reward,b_current_state)


        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        current_state  = state

        self.last_action = current_action
        self.last_state  = current_state

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward


def run_games(learner, hist, iters = 300, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
        
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 100, 10)

	# Save history. 
	np.save('hist',np.array(hist))


