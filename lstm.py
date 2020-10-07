import numpy as np
import random
import math

#! activation function hamoon :
def sigmoid(x): 
    return 1. / (1 + np.exp(-x))

def sigmoid_derivative(values): 
    return values * (1 - values)

def tanh_derivative(values): 
    return 1. - values ** 2

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        #! matrix vazn ha :
        """
        inja 4 ta vazn e (i, f, o, g) ro darim,
        ye tozih kotah bedam :
        
        1 - f : forget gate
        2 - i : input gate
        3 - o : output gate
        4 - g : Gate gate

        2 input az cell e ghabli darim ke behesh migim h(t - 1)
        va dovomish ham c(t - 1) hast ok ?
        va khodemoon too cell e h(t) hastim
        be in shekl ke ma 2 ta khorooji darim baraye har cell :
        h(t) va c(t)

        h(t) = o * tanh(c(t))
        c(t) = f * c(t - 1) + i * g

        """

        # inja vazn haro migirim :
        self.wg = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wi = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len) 
        self.wf = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        self.wo = rand_arr(-0.1, 0.1, mem_cell_ct, concat_len)
        
        # inja bias haro migirim : 
        self.bg = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bi = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bf = rand_arr(-0.1, 0.1, mem_cell_ct) 
        self.bo = rand_arr(-0.1, 0.1, mem_cell_ct) 
        
        # diffs (derivative of loss function w.r.t. all parameters)
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 