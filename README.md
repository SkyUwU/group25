# Use image to train AI to play claw machines
2024 Spring NYCU AI Final Project

# Task introduction
We used three methods to train AI to play claw machine.
  1. using the whole image as input and using DQN
  2. using the coordinates of the claw and the coordinates of the nearest item as input, and use DQN
  3. we transform the original problem into a picture classification problem and the model is trained using Teachable machine to decide the action using the pictures.

# Prerequisite
* Code environment
    * VS code
      
* Packages version
    * gym                        0.26.2
    * h5py                       3.8.0
    * keras                      2.11.0
    * numpy                      1.24.2
    * opencv-contrib-python      4.7.0.68
    * pandas                     1.5.3
    * PyAutoGUI                  0.9.54
    * tensorflow                 2.11.0

# Hyperparameters
We use DQN in method 1 & 2, and below are the hyperparameters
* learning_rate: 0.0002
* gamma: 0.97
* batch_size: 32
* epsilon changes according to agent.count
  * smaller than 2000: 0.02
  * between 2000 and 4000: 0.5
  * between 4000 and 7000: 0.8
  * greater than 7000: 0.95

# Experiment results
![image](https://github.com/SkyUwU/group25/assets/119331059/7855c73e-becd-4ae9-b348-479e6c36264d)

      
# Others
* Demo videos & the model of method 1
  * <https://drive.google.com/drive/folders/1xFnEnk5cSMM7lzEglypg7ZobP8tNhH-z>
