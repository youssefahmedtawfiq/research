# model/neuron.py
from brian2 import *

def get_izhikevich_eqs():
    # معادلات Izhikevich الرياضية
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
    du/dt = a*(b*v - u)/ms : 1
    I : 1
    a : 1
    b : 1
    c : 1
    d : 1
    '''
    threshold = 'v >= 30'
    reset = '''
    v = c
    u += d
    '''
    return eqs, threshold, reset

def get_state_params(state_num):
    # الـ 5 حالات الخاصة بخلايا Izhikevich
    params = {
        1: {'name': 'Regular Spiking (RS)',    'a': 0.02, 'b': 0.2,  'c': -65, 'd': 8},
        2: {'name': 'Intrinsically Bursting',  'a': 0.02, 'b': 0.2,  'c': -55, 'd': 4},
        3: {'name': 'Chattering (CH)',         'a': 0.02, 'b': 0.2,  'c': -50, 'd': 2},
        4: {'name': 'Fast Spiking (FS)',       'a': 0.1,  'b': 0.2,  'c': -65, 'd': 2},
        5: {'name': 'Low-Threshold Spiking',   'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2}
    }
    return params.get(state_num, params[1]) # Default is State 1