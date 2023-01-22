WORKING_DIR ="/home/Ravikumar/Developer/text_generation/"
DATA_DIR = WORKING_DIR + "dataset/"
DEFAULT_FILE_NAME = DATA_DIR + "joke.csv"
MAX_SEQ_LEN = 50
N_HIDDEN = 128
LR = 0.001
EMB_DIM = 32
BATCH_SIZE = 128

N_EPOCHS = 100
CHECKPOINTS_DIR = WORKING_DIR + f"checkpoints/batch_size_{BATCH_SIZE}-emb_dim_{EMB_DIM}-hidden_size_{N_HIDDEN}.pt"
NUM_LAYERS = 3
PATIENCE = 5
