import pickle
import time
import numpy as np
from .DQN_Agent import Double_DQNAgent
from .environment import environment
from .path_of_file import path_of_file

if __name__ == '__main__':

    # ------------------LOAD DATA------------------------
    with open("test_image.pickle", "rb") as f:
        dummy_image = pickle.load(f)

    dummy_image = dummy_image / np.amax(dummy_image)
    dummy_image_list = []
    dummy_image_list.append(dummy_image)
    dummy_landmark = [(113, 118, 193)]
    dummy_image_list_zipped = list(zip(dummy_image_list, dummy_landmark))
    # ----------------------------------------------------
    image = dummy_image_list_zipped[0]
    env = environment(image=dummy_image_list_zipped[0], patch_size=(45, 45, 45, 4), step_size=6)

    action_size = env.action_space
    state_size = env.patch_size


    agent = Double_DQNAgent(state_size, action_size, test_mode=True, enable_dist_estimator=False,
                            enable_dueling_dqn=True)


    agent.load(f'{path_of_file(__file__)}/Prio_Left_Kidneyweights_110000.h5')

    state = env.reset()
    agent.flush_buffer()

    env.render_image()
    timer = 0

    while (True):
        action, q_values = agent.act(state)

        if q_values >= 2.7 and q_values <= 4:
            env.deblur_image_first()

        if q_values < 2.7:
            env.deblur_image_second()

        next_state, reward, distance = env.step(action)
        # print("Estimated Distance: ", agent.estimate_distance()[0])
        print("Real Distance: ", distance)

        state = next_state

        env.render_patch(q_values=q_values, action=action)
        if timer == 0:
            time.sleep(5)
        timer = timer + 1
