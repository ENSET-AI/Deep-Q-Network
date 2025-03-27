# Game parameters
GRID_SIZE = 4
STATE_SIZE = GRID_SIZE * GRID_SIZE
ACTION_SIZE = 4 # Up, down, left, right
GAMMA = 0.9 # facteur de reduction
LEARNING_RATE = 0.01
EPSILON = 1.0 # Exploration initiale
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MEMORY_SIZE = 2000
EPISODES = 1000

# Déplacements possibles (Haut, Bas, Gauche, Droite)
MOVES = {
0: (1, 0), # Haut
1: (-1, 0), # Bas
2: (0, -1), # Gauche
3: (0, 1) # Droite
}