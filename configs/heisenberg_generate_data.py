from modelling import *
NAME = 'heisenberg_generate_data'

#-----------------DATASET--------------------
DATASET = '2Dheisenberg'
NUM_TRAIN = 80
NUM_TEST = 20
LATTICES = [2,12] 
EPSILON = 1/(LATTICES[0]*LATTICES[1])**2 # closeness of guided state and ground state
SHOTS = 1000 # number of sample for classical shadow

#-----------------DIRECTORY---------------------
TRAIN_DIR = "dataset/{}_{}_{}_train/".format(DATASET, LATTICES, SHOTS)
TEST_DIR = "dataset/{}_{}_{}_test/".format(DATASET, LATTICES, SHOTS)


