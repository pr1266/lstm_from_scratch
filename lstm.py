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

def rand_arr(a, b, *args):
    np.random.seed(0)
    return np.random.rand(*args) * (b - a) + a

class LstmParam:
    def __init__(self, mem_cell_ct, x_dim):
        self.mem_cell_ct = mem_cell_ct
        self.x_dim = x_dim
        concat_len = x_dim + mem_cell_ct

        #! matrix vazn ha :
        """
        inja 4 ta vazn e (i, f, o, g) ro darim,
        ye tozih kotah bedam :
        
        1 - f : Forget Gate 

        f(t) = sigmoid(W(f).[h(t - 1), x(t)] + Bf)

        be che dardi mikhore ?
        tensor e khorooji h(t - 1) momkene
        informationi toosh bashe ke ma niaz nadashte bashim
        va momken nakhaim ke in info tooye natije cell e ma
        tasir bezare ok ?
        dar vaghe ba activation e sigmoid miaim
        va ye halat e condition barash dar nazar migirim

        2 - i : Input Gate

        i(t) = sigmoid(W(i).[h(t - 1), x(t)] + Bi)
        c_hat(t) = tanh(W(c).[h(t - 1), x(t)] + Bc)

        in chi kar mikone ?
        sigmoid miad tasmim migire ke kodoom value ro update kone
        va tanh ham ye vector jadid misaze ke mitoone be state e 
        in cell i ke toosh hastim ezafe beshe

        3 - o : Output Gate

        kare in gate, tolid e 2 khorooji hast :
        h(t) va o(t) ke output e cell hast
        
        chetor ?
        be in shekl :

        o(t) = sigmoid(W(o).[h(t - 1), x(t)] + Bo)
        h(t) = o(t) * tanh(c(t))

        dar vaghe o(t) baraye tolid e h(t) bekar miad
        va jaii be onvan e khorooji estefade nemishe


        4 - g : Gate gate

        2 input az cell e ghabli darim ke behesh migim h(t - 1)
        va dovomish ham c(t - 1) hast ok ?
        va khodemoon too cell e h(t) hastim
        be in shekl ke ma 2 ta khorooji darim baraye har cell :
        h(t) va c(t)

        h(t) = o * tanh(c(t))
        c(t) = f * c(t - 1) + i * g


        khob ye nokte ro ham dar rabete ba back-propagation begam :
        inja ma tooye moshtagh giri, c(t) ro taghir midim na h(t) ro


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
        
        # inja ham moshtagh haro hesab mikonim ta tooye gradient descent estefade konim :
        self.wg_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wi_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wf_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.wo_diff = np.zeros((mem_cell_ct, concat_len)) 
        self.bg_diff = np.zeros(mem_cell_ct) 
        self.bi_diff = np.zeros(mem_cell_ct) 
        self.bf_diff = np.zeros(mem_cell_ct) 
        self.bo_diff = np.zeros(mem_cell_ct) 


    def apply_diff(self, lr = 1):

        """
        inja az moshtagh haii ke dar __init__ hesab kardim
        estefade mikonim baraye back propagation :
        """

        self.wg -= lr * self.wg_diff
        self.wi -= lr * self.wi_diff
        self.wf -= lr * self.wf_diff
        self.wo -= lr * self.wo_diff
        self.bg -= lr * self.bg_diff
        self.bi -= lr * self.bi_diff
        self.bf -= lr * self.bf_diff
        self.bo -= lr * self.bo_diff
        
        # moshtagharo sefr mikonim
        
        self.wg_diff = np.zeros_like(self.wg)
        self.wi_diff = np.zeros_like(self.wi) 
        self.wf_diff = np.zeros_like(self.wf) 
        self.wo_diff = np.zeros_like(self.wo) 
        self.bg_diff = np.zeros_like(self.bg)
        self.bi_diff = np.zeros_like(self.bi) 
        self.bf_diff = np.zeros_like(self.bf) 
        self.bo_diff = np.zeros_like(self.bo)


#! in class ham state haye lstm ro moshakhas mikone ke bala tar tozih dadam :
class LstmState:
    def __init__(self, mem_cell_ct, x_dim):
        self.g = np.zeros(mem_cell_ct)
        self.i = np.zeros(mem_cell_ct)
        self.f = np.zeros(mem_cell_ct)
        self.o = np.zeros(mem_cell_ct)
        self.s = np.zeros(mem_cell_ct)
        self.h = np.zeros(mem_cell_ct)
        self.bottom_diff_h = np.zeros_like(self.h)
        self.bottom_diff_s = np.zeros_like(self.s)

class LstmNode:
    def __init__(self, lstm_param, lstm_state):
        
        self.state = lstm_state
        self.param = lstm_param
        self.xc = None

    def bottom_data_is(self, x, s_prev = None, h_prev = None):
        """ 
        inja check mikonim ke aya in cell e ma ebtedaye zanjire
        network hast ya kheir
        age bood, miaim state e ghabli va
        h(t - 1) ro sefr(0) gharar midim :
        """
        if s_prev is None: s_prev = np.zeros_like(self.state.s)
        if h_prev is None: h_prev = np.zeros_like(self.state.h)
        
        # hala be har hal in state va h ro zakhire mikonim ta
        # estefade konim azash :
        self.s_prev = s_prev
        self.h_prev = h_prev

        # hala inja x(t) va h(t - 1) ro ba ham concat mikonim :
        xc = np.hstack((x,  h_prev))
        self.state.g = np.tanh(np.dot(self.param.wg, xc) + self.param.bg)
        self.state.i = sigmoid(np.dot(self.param.wi, xc) + self.param.bi)
        self.state.f = sigmoid(np.dot(self.param.wf, xc) + self.param.bf)
        self.state.o = sigmoid(np.dot(self.param.wo, xc) + self.param.bo)
        self.state.s = self.state.g * self.state.i + s_prev * self.state.f
        self.state.h = self.state.s * self.state.o

        self.xc = xc
    
    def top_diff_is(self, top_diff_h, top_diff_s):
        
        # back prop ro anjam midim :
        ds = self.state.o * top_diff_h + top_diff_s
        do = self.state.s * top_diff_h
        di = self.state.g * ds
        dg = self.state.i * ds
        df = self.s_prev * ds

        di_input = sigmoid_derivative(self.state.i) * di 
        df_input = sigmoid_derivative(self.state.f) * df 
        do_input = sigmoid_derivative(self.state.o) * do 
        dg_input = tanh_derivative(self.state.g) * dg

        self.param.wi_diff += np.outer(di_input, self.xc)
        self.param.wf_diff += np.outer(df_input, self.xc)
        self.param.wo_diff += np.outer(do_input, self.xc)
        self.param.wg_diff += np.outer(dg_input, self.xc)
        self.param.bi_diff += di_input
        self.param.bf_diff += df_input       
        self.param.bo_diff += do_input
        self.param.bg_diff += dg_input       

        dxc = np.zeros_like(self.xc)
        dxc += np.dot(self.param.wi.T, di_input)
        dxc += np.dot(self.param.wf.T, df_input)
        dxc += np.dot(self.param.wo.T, do_input)
        dxc += np.dot(self.param.wg.T, dg_input)

        self.state.bottom_diff_s = ds * self.state.f
        self.state.bottom_diff_h = dxc[self.param.x_dim:]

class LstmNetwork():
    def __init__(self, lstm_param):
        self.lstm_param = lstm_param
        self.lstm_node_list = []
        # input sequence ro midim behesh :
        self.x_list = []

    def y_list_is(self, y_list, loss_layer):
        """
        hala bar asas e target sequence ke dadim behesh mitoonim vazn hasho 
        taghir bedim toye loss layer
        """
        assert len(y_list) == len(self.x_list)
        idx = len(self.x_list) - 1
        # first node only gets diffs from label ...
        loss = loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
        diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
        # inja chon s ro baraye node e aval sefr kardim tasiri roo h(t + 1) nadare
        diff_s = np.zeros(self.lstm_param.mem_cell_ct)
        self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
        idx -= 1

        while idx >= 0:
            loss += loss_layer.loss(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h = loss_layer.bottom_diff(self.lstm_node_list[idx].state.h, y_list[idx])
            diff_h += self.lstm_node_list[idx + 1].state.bottom_diff_h
            diff_s = self.lstm_node_list[idx + 1].state.bottom_diff_s
            self.lstm_node_list[idx].top_diff_is(diff_h, diff_s)
            idx -= 1 

        return loss

    def x_list_clear(self):
        self.x_list = []

    def x_list_add(self, x):
        self.x_list.append(x)
        if len(self.x_list) > len(self.lstm_node_list):
            # inja node ezafe mikonim :
            lstm_state = LstmState(self.lstm_param.mem_cell_ct, self.lstm_param.x_dim)
            self.lstm_node_list.append(LstmNode(self.lstm_param, lstm_state))

        idx = len(self.x_list) - 1
        if idx == 0:

            self.lstm_node_list[idx].bottom_data_is(x)
        else:

            s_prev = self.lstm_node_list[idx - 1].state.s
            h_prev = self.lstm_node_list[idx - 1].state.h
            self.lstm_node_list[idx].bottom_data_is(x, s_prev, h_prev)