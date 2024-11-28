# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example use of the CFR algorithm on Kuhn Poker."""

import pickle
import sys
from absl import app
from absl import flags
from tqdm import tqdm
from open_spiel.python.algorithms import discounted_cfr
from open_spiel.python.algorithms import exploitability

import matplotlib.pyplot as plt
import numpy as np

import pyspiel

universal_poker = pyspiel.universal_poker

FLAGS = flags.FLAGS

flags.DEFINE_enum("solver", "dcfr", ["cfr", "dcfr", "cfrplus"], "CFR solver")
_ITERATIONS = flags.DEFINE_integer("iterations", 10, "Number of iterations")

CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF = """\
GAMEDEF
limit
numPlayers = 2
numRounds = 2
blind = 5 0
raiseSize = 10 20 20
maxRaises = 6 6 
numSuits = 2
numRanks = 6
numHoleCards = 1
numBoardCards = 0 1
stack = 100
END GAMEDEF
"""
# game_def = {
#     'numPlayers': 2,
#     'numRounds': 3,
#     'numSuits': 4,
#     'numRanks': 3,
#     'numHoleCards': 1,
#     'numBoardCards': [0, 1, 1],
#     'maxRaises': [1, 1,1],
#     'raiseSize': [2, 4],
#     'stack': 100
# }

def plot_exploitability(exploitabilities):
    plt.figure(figsize=(10, 6))
    plt.plot(exploitabilities, label='Exploitability')
    plt.xlabel('Iteration')
    plt.ylabel('Exploitability')
    plt.yscale('log')
    plt.title('Exploitability over Iterations')
    plt.legend()
    plt.show()

def main(_):
  game = universal_poker.load_universal_poker_from_acpc_gamedef(
      CUSTOM_LIMIT_HOLDEM_ACPC_GAMEDEF
  )

  solver = None
  if FLAGS.solver == "cfr":
    solver = pyspiel.CFRSolver(game)
  elif FLAGS.solver == "dcfr":
    solver = discounted_cfr.DCFRSolver(game)
  elif FLAGS.solver == "cfrplus":
    solver = pyspiel.CFRPlusSolver(game)
  elif FLAGS.solver == "cfrbr":
    solver = pyspiel.CFRBRSolver(game)
  else:
    print("Unknown solver")
    sys.exit(0)

  exploitabilities = []

  for i in tqdm(iterable=range(int(_ITERATIONS.value / 2)), total=int(_ITERATIONS.value / 2),):
    solver.evaluate_and_update_policy()
    exploit = exploitability.exploitability(game, solver.average_policy())
    exploitabilities.append(exploit)
    # print("Iteration {} exploitability: {:.6f}".format(i, exploit))

  filename = "/tmp/{}_solver.pickle".format(FLAGS.solver)
  print("Persisting the model...")
  with open(filename, "wb") as file:
    pickle.dump(solver, file, pickle.HIGHEST_PROTOCOL)

  print("Loading the model...")
  with open(filename, "rb") as file:
    loaded_solver = pickle.load(file)
  print("Exploitability of the loaded model: {:.6f}".format(
      exploitability.exploitability(game, solver.average_policy())))

  for i in tqdm(iterable=range(int(_ITERATIONS.value / 2)), total=int(_ITERATIONS.value / 2),):
    loaded_solver.evaluate_and_update_policy()
    exploit = exploitability.exploitability(game, solver.average_policy())
    exploitabilities.append(exploit)
    tabular_policy = loaded_solver.tabular_average_policy()
    # print(f"Tabular policy length: {len(tabular_policy)}")
    # print(
    #     "Iteration {} exploitability: {:.6f}".format(
    #         int(_ITERATIONS.value / 2) + i,
    #         exploit,
    #     )
    # )

  plot_exploitability(exploitabilities)

if __name__ == "__main__":
  app.run(main)
