import ale_python_interface
import numpy as np

# Init ale
ale = ale_python_interface.ALEInterface()
#ale.setInt('random_seed', 123) TODO: find out what this is

ale.setBool('display_screen', True)
ale.setFloat('repeat_action_probability', 0)

full_rom_path = "../roms/breakout.bin"
ale.loadROM(full_rom_path)

# testing ale
legal_actions = ale.getMinimalActionSet()
print(legal_actions)
total_reward = 0

repeat_action = 20
a_count = 0
index = 0

while not ale.game_over():

    if a_count > repeat_action:
        index = (index + 1) % len(legal_actions)
        a_count = 0

    a = legal_actions[index]

    print "Action: ", a
    a_count += 1

    # a = np.random.choice(legal_actions, 1)[0]
    reward = ale.act(a)
    total_reward += reward

print("Episode ended with score: ", total_reward)