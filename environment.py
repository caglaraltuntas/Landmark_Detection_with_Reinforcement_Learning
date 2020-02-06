import math
import random
from collections import deque
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter


class environment():

    def __init__(self, image, patch_size, step_size, BlurMode=True):

        self.image = image[0]
        self.original = self.image
        self.blurred_image_level1 = gaussian_filter(self.original, sigma=1)
        self.blurred_image_level2 = gaussian_filter(self.original, sigma=2)
        self.image = self.blurred_image_level2

        self.fig = 0
        self.ax1 = 0
        self.ax2 = 0
        self.ax3 = 0
        self.ax4 = 0
        self.average_q_values = []
        self.real_distance_list = []
        # -----------------------------------------------------------------------------

        self.action_space = 6

        self.limit_bottom = self.image.shape[0] - 1 - (patch_size[0] - 1) / 2
        self.limit_top = (patch_size[0] - 1) / 2

        self.limit_right = self.image.shape[1] - 1 - (patch_size[1] - 1) / 2
        self.limit_left = (patch_size[1] - 1) / 2

        self.limit_front = self.image.shape[2] - 1 - (patch_size[2] - 1) / 2
        self.limit_back = (patch_size[2] - 1) / 2
        # -----------------------------------------------------------------------------

        self.patch_center = [random.randint(self.limit_top, self.limit_bottom),
                             random.randint(self.limit_left, self.limit_right),
                             random.randint(self.limit_back, self.limit_front)]

        self.patch_size = patch_size
        self.step_size = step_size
        self.landmark_center = image[1]
        self.distances = deque(maxlen=2)
        self.dist = math.sqrt((self.landmark_center[0] - self.patch_center[0]) ** 2 +
                              (self.landmark_center[1] - self.patch_center[1]) ** 2 +
                              (self.landmark_center[2] - self.patch_center[2]) ** 2)

        self.distances.append(self.dist)

    def patch_show(self):  # Gives you the state
        self.patch = self.image[(self.patch_center[0] - int((self.patch_size[0] - 1) / 2)):(
                self.patch_center[0] + int((self.patch_size[0] - 1) / 2 + 1)),
                     (self.patch_center[1] - int((self.patch_size[1] - 1) / 2)):(
                             self.patch_center[1] + int((self.patch_size[1] - 1) / 2 + 1)),
                     (self.patch_center[2] - int((self.patch_size[2] - 1) / 2)):(
                             self.patch_center[2] + int((self.patch_size[2] - 1) / 2 + 1))]

        return self.patch

    def step(self, direction):

        # R:0, L:1, U:2, D:3, F:4, B:5,
        if direction == 0:
            if self.patch_center[1] + self.step_size <= self.limit_right:  # +y
                self.patch_center[1] = self.patch_center[1] + self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)

        elif direction == 1:
            if self.patch_center[1] - self.step_size >= self.limit_left:
                self.patch_center[1] = self.patch_center[1] - self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)

        elif direction == 2:
            if self.patch_center[0] - self.step_size >= self.limit_top:
                self.patch_center[0] = self.patch_center[0] - self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)

        elif direction == 3:
            if self.patch_center[0] + self.step_size <= self.limit_bottom:  # +x
                self.patch_center[0] = self.patch_center[0] + self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)

        elif direction == 4:
            if self.patch_center[2] + self.step_size <= self.limit_front:  # +z
                self.patch_center[2] = self.patch_center[2] + self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)

        elif direction == 5:
            if self.patch_center[2] - self.step_size >= self.limit_back:
                self.patch_center[2] = self.patch_center[2] - self.step_size
            else:
                self.distance()
                return (self.patch_show(), -1, self.dist)


        self.distance()

        return (self.patch_show(), self.reward(), self.dist)

    def distance(self):

        self.dist = math.sqrt((self.landmark_center[0] - self.patch_center[0]) ** 2 +
                              (self.landmark_center[1] - self.patch_center[1]) ** 2 +
                              (self.landmark_center[2] - self.patch_center[2]) ** 2)

        self.distances.append(self.dist)

    def reward(self):

        return (self.distances[0] - self.distances[1]) / self.step_size

    def reset(self):
        #self.patch_center = [25, 25, 280]
        self.patch_center = [random.randint(self.limit_top, self.limit_bottom),
                             random.randint(self.limit_left, self.limit_right),
                             random.randint(self.limit_back, self.limit_front)]

        self.distances = deque(maxlen=2)
        self.dist = math.sqrt((self.landmark_center[0] - self.patch_center[0]) ** 2 +
                              (self.landmark_center[1] - self.patch_center[1]) ** 2 +
                              (self.landmark_center[2] - self.patch_center[2]) ** 2)

        self.distances.append(self.dist)
        # self.image = self.blurred_image_level2
        self.step_size = 4

        return self.patch_show()

    def render_image(self):

        # ---------------Image sliced from x direction--------
        landmark_x = self.landmark_center[0]
        image_slice_x = np.squeeze(self.original[landmark_x:landmark_x + 1, :, :], axis=0)
        image_slice_x = np.rot90(image_slice_x)  # slices are wrong instead pass the coordinate of the landmark

        # ---------------Image sliced from y direction--------
        landmark_y = self.landmark_center[1]
        image_slice_y = np.squeeze(self.original[:, landmark_y:landmark_y + 1, :], axis=1)
        image_slice_y = np.rot90(image_slice_y)

        # ---------------Show the landmark on both slices-------

        image_slice_y[((315 - self.landmark_center[2]) - 2):((315 - self.landmark_center[2]) + 2),
        (self.landmark_center[0] - 2):(self.landmark_center[0] + 2)] = 0

        image_slice_x[((315 - self.landmark_center[2]) - 2):((315 - self.landmark_center[2]) + 2),
        (self.landmark_center[1] - 2):(self.landmark_center[1] + 2)] = 0

        # ---------------------Show image------------------------

        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(9, 9))

        self.ax1.imshow(image_slice_y, cmap="gray")
        self.ax2.imshow(image_slice_x, cmap="gray")
        self.ax1.set_title("Image Slice from Frontal Plane")
        self.ax2.set_title("Image Slice from Sagittal Plane")
        self.ax3.set_title("Average Q Values")

        self.ax3.set_ylabel("Average Q Value")
        self.ax4.set_title("Distances")

        self.ax4.set_ylabel("Real Distance(Pixel-Wise)")

    def render_patch(self, q_values, action):

        Q_values = list(np.around(np.array(q_values), 2))
        self.average_q_values.append(np.mean(Q_values))
        # self.real_distance_list.append(self.dist)
        # image shape is (320, 260, 316) : x, y, z

        #  clear_output(wait=True)

        rec_first_value_x = self.patch_center[1] - (self.patch_size[1] - 1) / 2  # y
        rec_second_value_x = (315 - self.patch_center[2] - (self.patch_size[2] - 1) / 2)  # z

        rec_first_value_y = self.patch_center[0] - (self.patch_size[0] - 1) / 2  # x
        rec_second_value_y = (315 - self.patch_center[2] - (self.patch_size[2] - 1) / 2)  # z
        # -----------------------------------------


        rect_y = Rectangle((rec_first_value_y, rec_second_value_y), self.patch_size[2], self.patch_size[0], linewidth=1,
                           edgecolor='r', facecolor='none')
        dot_y = Rectangle((self.patch_center[0], 315 - self.patch_center[2]), 2, 2, linewidth=1, edgecolor='r',
                          facecolor='red')

        rect_x = Rectangle((rec_first_value_x, rec_second_value_x), self.patch_size[2], self.patch_size[1], linewidth=1,
                           edgecolor='r', facecolor='none')
        dot_x = Rectangle((self.patch_center[1], 315 - self.patch_center[2]), 2, 2, linewidth=1, edgecolor='r',
                          facecolor='red')

        self.ax1.add_patch(rect_y)
        self.ax1.add_patch(dot_y)
        self.ax2.add_patch(rect_x)
        self.ax2.add_patch(dot_x)
        self.ax3.plot(self.average_q_values, color="blue")

        objects = ('Front', 'Back', 'Left', 'Right', 'Up', 'Down')
        x_pos = np.arange(len(objects))

        # colors = ['r', 'b', 'b', 'b', 'b', 'b']

        colors = ["r" if i == action else "b" for i in range(self.action_space)]

        self.ax4.bar(x_pos, Q_values, align='center', alpha=0.8, color=colors)
        self.ax4.set_xticks(x_pos)
        self.ax4.set_xticklabels(objects)
        self.ax4.set_xlabel("Possible Actions\n\nInstant Q-Values: {}".format(Q_values))
        self.ax3.set_xlabel("Total Number of Actions\n\nInstant Distance(mm):{:0.2f}".format(self.dist))
        plt.pause(0.000001)

        [p.remove() for p in reversed(self.ax1.patches)]
        [p.remove() for p in reversed(self.ax2.patches)]
        self.ax4.clear()

    def deblur_image_first(self):
        self.image = self.blurred_image_level1
        self.step_size = 3

    def deblur_image_second(self):
        self.image = self.original
        self.step_size = 1

    def blur_image_initial(self):
        self.image = self.blurred_image_level2
        self.step_size = 6
