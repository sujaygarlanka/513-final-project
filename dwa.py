import numpy as np

"""
Fetch Robot 

Max velocity = 1.0 m/s
Webpage: https://fetchrobotics.com/fetch-mobile-manipulator/
White paper on robot: https://fetch3.wpenginepowered.com/wp-content/uploads/2021/06/Fetch-and-Freight-Workshop-Paper.pdf


"""

"""
DWA

Paper: https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf
Youtube video with algorithm: https://www.youtube.com/watch?v=tNtUgMBCh2g&t=1s
Youtube video demoing it with attached code: https://www.youtube.com/watch?v=Mdg9ElewwA0

"""

class DWA():
    def __init__(self, env):
        self.env = env
        self.counter = 0
        self.toggle = True

    # action map for Fetch robot - 11 controls
    # 0 - forward/back
    # 1 - rotation
    def get_next_action(self, state, destination):
        x = state['proprioception'][0]
        y = state['proprioception'][1]
        action = np.zeros((11,))
        if self.counter == 100:
            self.toggle = not self.toggle
            self.counter = 0
        if self.toggle:        
            # action[0] = 0.0
            action[1] = 1.0
        else:
            # action[0] = 1.0
            action[0] = 0.0

        self.counter += 1
        return action