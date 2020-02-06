import os
def path_of_file(file):
    current_path = os.path.realpath(__file__)
    current_path = os.path.dirname(current_path)

    return current_path
