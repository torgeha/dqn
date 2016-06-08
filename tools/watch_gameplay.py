"""
The starting point for a training session with NATURE parameters.
"""

import launcher
import sys

class DefaultParams:
    """
    Static class containing all default parameters for the training.
    """

    #NATURE
     # TODO: to watch a trained network -->
    # steps per epoch = 0
    # steps per test = 10000
    # display_screen = true
    # specify pickle_file path

    # Experiment Parameters
    # ----------------------
    # STEPS_PER_EPOCH = 250000
    # EPOCHS = 200
    # STEPS_PER_TEST = 125000

    # To watch
    STEPS_PER_EPOCH = 0
    EPOCHS = 200
    STEPS_PER_TEST = 125000

    # ----------------------
    # ALE Parameters
    # ----------------------
    ROM_PATH = "../roms/"
    ROM_NAME = 'breakout.bin'
    FRAME_SKIP = 4
    REPEAT_ACTION_PROBABILITY = 0
    # DISPLAY_SCREEN = False
    DISPLAY_SCREEN = True

    # ----------------------
    # Agent/Network parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = .00025
    DISCOUNT = .99
    RMS_DECAY = .95 # (Rho)
    RMS_EPSILON = .01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 1000000
    PHI_LENGTH = 4
    UPDATE_FREQUENCY = 4
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "nature_cuda"
    FREEZE_INTERVAL = 10000
    REPLAY_START_SIZE = 50000
    RESIZE_METHOD = 'scale'
    RESIZED_WIDTH = 84
    RESIZED_HEIGHT = 84
    DEATH_ENDS_EPISODE = 'true'
    MAX_START_NULLOPS = 30
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False

    if not sys.argv[1]:
        sys.exit(1)

    NETWORK_PICKLE_FILE = sys.argv[1]
    # NETWORK_PICKLE_FILE = None
    EXPERIMENT_PREFIX = "nature-WATCH"
    RANDOM_NETWORK_PICKLE = False



if __name__ == "__main__":
    launcher.start_training(DefaultParams)

