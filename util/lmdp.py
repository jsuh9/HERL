from enum import Enum
from typing import Tuple
import numpy as np


class StateType(Enum):
    NORMAL = 0
    GOAL = 1
    TRAP = 2

class GridWorldMDP:
    def __init__(self, size: int = 4, 
                 goal_state: Tuple[int, int] = (3, 3), # pre-speicied goal state assumign 4-by-4 grid
                 trap_state: Tuple[int, int] = (3, 0), # pre-speicied trap state assumign 4-by-4 grid
                 lambda_reg = 1.0):

        self.size = size
        self.goal_state = goal_state
        self.trap_state = trap_state
        
        # 9 actions: 8 directions (with diag) + stay
        self.actions = [
            (0, 0),   # stay
            (0, 1),   # right
            (1, 1),   # down-right
            (1, 0),   # down
            (1, -1),  # down-left
            (0, -1),  # left
            (-1, -1), # up-left
            (-1, 0),  # up
            (-1, 1)   # up-right   
        ]

        self.lambda_reg = lambda_reg

        # create state space (excluding absorbing states)
        self.states = [(i, j) for i in range(size) for j in range(size) 
                      if (i, j) != goal_state and (i, j) != trap_state]
        self.state_to_idx = {state: idx for idx, state in enumerate(self.states)}
        self.n_states = len(self.states)
    
    def get_valid_actions(self, state):
        valid_actions = []
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            if next_state != self.trap_state:
                valid_actions.append(action)
        return valid_actions
    
    def get_next_state(self, state, action):
        x, y = state
        dx, dy = action
        next_x, next_y = x + dx, y + dy
        
        if (next_x, next_y) == self.trap_state:
            return state
        elif 0 <= next_x < self.size and 0 <= next_y < self.size:
            return (next_x, next_y)
        return state
    
    def cost(self, state):
        # cost function (accoding to Todorov's): q > 0 everywhere except goal
        if state == self.goal_state:
            return 0.0
        return 0.5
    
    def behavioral_policy(self, action, state):
        valid_actions = self.get_valid_actions(state)
        return 1.0 / len(valid_actions) if action in valid_actions else 0.0
    
    def construct_A_w(self):
        A = np.zeros((self.n_states, self.n_states))
        w = np.zeros(self.n_states)
        for i, state in enumerate(self.states):
            valid_actions = self.get_valid_actions(state)
            for action in valid_actions:
                next_state = self.get_next_state(state, action)
                # uniform probability over valid actions
                term = (1.0 / len(valid_actions)) * np.exp(-self.cost(state) / self.lambda_reg) 
                if next_state == self.goal_state:
                    w[i] += term
                elif next_state != state:  # valid transition to normal state
                    j = self.state_to_idx[next_state]
                    A[i, j] += term
        return A, w
    
    def solve_Z_star(self):
        A, w = self.construct_A_w()
        I = np.eye(self.n_states)
        Z_star_vector = np.linalg.solve(I - A, w)
        Z_star = {state: Z_star_vector[i] for i, state in enumerate(self.states)}
        Z_star[self.goal_state] = 1.0 
        Z_star[self.trap_state] = 0.0
        return Z_star
    
    def solve_Z_star_vi(self, max_iter: int = 100, tolerance: float = 1e-8):
        A, w = self.construct_A_w()
        Z = np.ones(self.n_states)
        for iter in range(max_iter):
            # update Z using Z = AZ + w
            Z_new = A @ Z + w
            max_diff = np.max(np.abs(Z_new - Z))
            Z = Z_new
            if max_diff < tolerance:
                print(f"VI converged; took {iter+1} iterations")
                break
        if iter == max_iter - 1:
            print("VI not converged within max iterations")
        
        Z_star = {state: Z[i] for i, state in enumerate(self.states)}
        Z_star[self.goal_state] = 1.0
        Z_star[self.trap_state] = 0.0
        
        return Z_star