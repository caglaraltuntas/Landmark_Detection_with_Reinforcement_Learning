import pickle
import numpy as np
import tensorflow as tf
from .A2C_Agent import A2C_Agent
from .environment import environment
from .parse import parse_function

if __name__ == '__main__':

    filename = "composed_w.tfrecord"
    dataset = tf.data.TFRecordDataset(filename).map(parse_function)
    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()

    with tf.Session() as sess:
        try:

            entire_image = sess.run(image)

            single_slice_x = entire_image[160:161, :, :]
            single_slice_y = entire_image[:, 130:131, :]  # x, z, y
            single_slice_z = entire_image[:, :, 290:291]

            array_2d_x = np.squeeze(single_slice_x, axis=0)
            array_2d_y = np.squeeze(single_slice_y, axis=1)
            array_2d_z = np.squeeze(single_slice_z, axis=2)

            with open("entire_image.pickle", "wb") as pickle_out:
                pickle.dump(entire_image, pickle_out)

        except tf.errors.OutOfRangeError:
            pass

    # ------------------LOAD DATA------------------------

    with open("entire_image.pickle", "rb") as f:
        dummy_image = pickle.load(f)

    dummy_image = dummy_image / np.amax(dummy_image)
    dummy_image_list = []
    dummy_image_list.append(dummy_image)
    dummy_landmark = [(193, 100, 197)]
    dummy_image_list_zipped = list(zip(dummy_image_list, dummy_landmark))
    # ----------------------------------------------------

    env = environment(image=dummy_image_list_zipped[0], patch_size=(45, 45, 45, 4), step_size=6)

    action_size = env.action_space
    state_size = env.patch_size

    # agent = Double_DQNAgent(state_size, action_size, enable_dist_estimator=False, test_mode=False,
    # enable_dueling_dqn = False, enable_prioritized_replay=False)
    agent = A2C_Agent(state_size, action_size, test_mode=False)
    batch_size = 32
    n_episodes = 150
    update_frequency = 50
    # prioritized_replay = agent.enable_prioritized_replay

    output_dir = "Model_3D_Localization"
    output_dir_dist_estimation = "Model_3D_Localization_dist_estimation"
    total_time = 0
    image_index = 0
    estimated_distance = 0.0
    real_distance = 0.0

    for image in dummy_image_list_zipped:  # A2C method
        env = environment(image=image, patch_size=(45, 45, 45, 4), step_size=6)
        for e in range(n_episodes):
            agent.flush_buffer()
            state = env.reset()
            # flag  = 0
            for time in range(151):  # max time, increase this number later

                if env.dist > 15:
                    env.blur_image_initial()
                if env.dist >= 5 and env.dist <= 15:
                    env.deblur_image_first()
                if env.dist < 5:
                    env.deblur_image_second()

                # env.render()
                action, policy_values = agent.act(state)
                next_state, reward, distance = env.step(action)  # real distance

                agent.replay(state, action, reward, next_state)
                state = next_state

                print("{} th image is being trained".format(image_index))
                print("Total Time: ", total_time)
                print("Episode: ", e)
                print("Real Distance: ", distance)
                print("Policy Vector: ", policy_values)
                print("Action: ", action)
                print("Reward", reward)
                print("Epsilon", agent.epsilon)

                total_time += 1

        image_index += 1
