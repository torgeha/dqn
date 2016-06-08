"""
This script starts the training process. Called by run.py where paramteters are determined.
"""

import os
import ale_python_interface
import logging
import numpy as np
import time
import theano
import cPickle
from logging.handlers import RotatingFileHandler

import q_network
import ale_agent
import ale_experiment

def start_training(params):
    """
    Initialize rom, game, agent, network and start a training run
    """



    # CREATE A FOLDER TO HOLD RESULTS


    exp_pref = "../results/" + params.EXPERIMENT_PREFIX
    time_str = time.strftime("_%m-%d-%H-%M_", time.gmtime())
    exp_dir = exp_pref + time_str + \
                   "{}".format(params.LEARNING_RATE).replace(".", "p") + "_" \
                   + "{}".format(params.DISCOUNT).replace(".", "p")

    try:
        os.stat(exp_dir)
    except OSError:
        os.makedirs(exp_dir)

    logger = logging.getLogger("DeepLogger")
    logger.setLevel(logging.INFO)

    # Logging filehandler
    #fh = logging.FileHandler(exp_dir + "/log.log")
    # Rotate file when filesize is 5 mb
    fh = RotatingFileHandler(exp_dir + "/log.log", maxBytes=5000000, backupCount=100)

    fh.setLevel(logging.INFO)

    # Console filehandler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)

    # Prevent nohup from producing large log file, logging to file is handled internally
    # logger.addHandler(ch)

    log_params(logger, params)

    #logging.basicConfig(level=logging.INFO, filename=exp_dir + "/log.log")


    if params.DETERMINISTIC:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()

    if params.CUDNN_DETERMINISTIC:
        theano.config.dnn.conv.algo_bwd = 'deterministic'

    # Init ale
    ale = ale_python_interface.ALEInterface()
    ale.setInt('random_seed', 123)
    ale.setBool('display_screen', params.DISPLAY_SCREEN)
    ale.setFloat('repeat_action_probability', params.REPEAT_ACTION_PROBABILITY)
    full_rom_path = os.path.join(params.ROM_PATH, params.ROM_NAME)
    ale.loadROM(full_rom_path)
    num_actions = len(ale.getMinimalActionSet())

    print "Legal actions: ", num_actions
    print ale.getMinimalActionSet()

    # Instantiate network
    logger.info("Setting up network...")
    network = None # Be able to continue training from a network or watch a network play
    if (params.NETWORK_PICKLE_FILE is None):
        logger.info("Initializing a new random network...")
        network = q_network.DeepQLearner(params.RESIZED_WIDTH,
                                             params.RESIZED_HEIGHT,
                                             num_actions,
                                             params.PHI_LENGTH,
                                             params.DISCOUNT,
                                             params.LEARNING_RATE,
                                             params.RMS_DECAY,
                                             params.RMS_EPSILON,
                                             params.MOMENTUM,
                                             params.CLIP_DELTA,
                                             params.FREEZE_INTERVAL,
                                             params.BATCH_SIZE,
                                             params.NETWORK_TYPE,
                                             params.UPDATE_RULE,
                                             params.BATCH_ACCUMULATOR,
                                             rng)
    else:
        logger.info("Loading network instance from file...")
        handle = open(params.NETWORK_PICKLE_FILE, 'r')
        network = cPickle.load(handle)


    # Only used when getting a random network
    if params.RANDOM_NETWORK_PICKLE:
        import sys
        sys.setrecursionlimit(10000)
        result_net_file = open(params.EXPERIMENT_PREFIX + '.pkl', 'w')
        print "File opened"
        cPickle.dump(network, result_net_file, -1)
        print "Pickle dumped"
        result_net_file.close()
        sys.exit(0)


    # Instatiate agent
    logger.info("Setting up agent...")
    agent = ale_agent.NeuralAgent(network,
                                  params.EPSILON_START,
                                  params.EPSILON_MIN,
                                  params.EPSILON_DECAY,
                                  params.REPLAY_MEMORY_SIZE,
                                  exp_dir,
                                  params.REPLAY_START_SIZE,
                                  params.UPDATE_FREQUENCY,
                                  rng)

    # Instantiate experient
    logger.info("Setting up experiment...")
    experiment = ale_experiment.ALEExperiment(ale, agent,
                                              params.RESIZED_WIDTH,
                                              params.RESIZED_HEIGHT,
                                              params.RESIZE_METHOD,
                                              params.EPOCHS,
                                              params.STEPS_PER_EPOCH,
                                              params.STEPS_PER_TEST,
                                              params.FRAME_SKIP,
                                              params.DEATH_ENDS_EPISODE,
                                              params.MAX_START_NULLOPS,
                                              rng)


    # Run experiment
    logger.info("Running experiment...")
    experiment.run()

def log_params(log, p):
    """
    log all params
    """

    # Experiment params
    log.info("=============== Experiment Params ===============")
    log.info("Epochs: " + str(p.EPOCHS))
    log.info("Steps per epoch: " + str(p.STEPS_PER_EPOCH))
    log.info("Steps per test: " + str(p.STEPS_PER_TEST))

    # Ale params
    log.info("\n ================= ALE params ==================")
    log.info("ROM path: " + str(p.ROM_PATH))
    log.info("ROM name: " + str(p.ROM_NAME))
    log.info("Frame skip: " + str(p.FRAME_SKIP))
    log.info("Diplay screen: " + str(p.DISPLAY_SCREEN))
    log.info("Repeat action probability: " + str(p.REPEAT_ACTION_PROBABILITY))

    # Network params
    log.info("\n ================ Network params =============== ")
    log.info("Update rule: " + str(p.UPDATE_RULE))
    log.info("Batch accumulator: " + str(p.BATCH_ACCUMULATOR))
    log.info("Learning rate: " + str(p.LEARNING_RATE))
    log.info("Discount: " + str(p.DISCOUNT))
    log.info("RMS decay (Rho): " + str(p.RMS_DECAY)) # (Rho)
    log.info("RMS epsilon: " + str(p.RMS_EPSILON))
    log.info("Momentum: " + str(p.MOMENTUM))
    log.info("Clip delta: " + str(p.CLIP_DELTA))
    log.info("Epsilon start: " + str(p.EPSILON_START))
    log.info("Epsilon min: " + str(p.EPSILON_MIN))
    log.info("Epsilon decay: " + str(p.EPSILON_DECAY))
    log.info("Phi length: " + str(p.PHI_LENGTH))
    log.info("Update frequency: " + str(p.UPDATE_FREQUENCY))
    log.info("Replay memory size: " + str(p.REPLAY_MEMORY_SIZE))
    log.info("Batch size: " + str(p.BATCH_SIZE))
    #NETWORK_TYPE = "nips_dnn"
    log.info("Network type: " + str(p.NETWORK_TYPE))
    log.info("Freeze interval: " + str(p.FREEZE_INTERVAL))
    log.info("Replay start size: " + str(p.REPLAY_START_SIZE))
    log.info("Resize method: " + str(p.RESIZE_METHOD))
    log.info("Resize width: " + str(p.RESIZED_WIDTH))
    log.info("Resize height: " + str(p.RESIZED_HEIGHT))
    log.info("Death ends episode: " + str(p.DEATH_ENDS_EPISODE))
    log.info("Max start nullops: " + str(p.MAX_START_NULLOPS))
    log.info("Deterministic: " + str(p.DETERMINISTIC))
    log.info("CUDNN deterministic: " + str(p.CUDNN_DETERMINISTIC))
    log.info("Experiment prefix: " + str(p.EXPERIMENT_PREFIX))
    log.info("=====================================================")
