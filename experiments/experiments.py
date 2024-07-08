import argparse

import sys
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/utils')
sys.path.insert(0, 'C:/Users/aybuk/Git/causal-bandits/src/algorithms')
from src.utils import MAB, SCM, graph_generator, plots
from src.algorithms import thompson, causal_thompson, ucb
