from IPython import display as ipythondisplay
import gym
from gym import ObservationWrapper, spaces
from gym.wrappers import Monitor
import io
import glob
import base64
from IPython.display import HTML
import numpy as np

"""
Utility functions to enable video recording of gym environment and displaying it
To enable video, just do "env = wrap_env(env)""
"""

def show_videos():
    """
    Shows videos from the `video` folder inside the Jupyter notebook.
    """

    mp4list = glob.glob('video/*.mp4')
    mp4list.sort()
    for mp4 in mp4list:
        print(f"\nSHOWING VIDEO {mp4}")
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''<video alt="test" autoplay 
                    loop controls style="height: 400px;">
                    <source src="data:video/mp4;base64,{0}" type="video/mp4" />
                    </video>'''.format(encoded.decode('ascii'))))
    
def wrap_env(env, video_callable=None):
    """
    When running the environment, periodically save videos of it in the `video` folder.
    """

    env = Monitor(env, './video', force=True, video_callable=video_callable)
    return env 

  
class CartpolePixels(ObservationWrapper):

    def __init__(self, env : "gym.Env"):
        """Wrapper for the CartPole environment that outputs
        pixel images as observations instead of the state vector.

        Parameters
        ----------
        env : gym.Env
            Instance of CartPole environment
        """

        super().__init__(env)
        self.env = env

        #---Initialize pixel transform---#

        #(Code adapted from the render() method in cartpole.py:
        #https://github.com/openai/gym/blob/a5a6ae6bc0a5cfc0ff1ce9be723d59593c165022/gym/envs/classic_control/cartpole.py)

        self.width  = 600 #Screen returned by env.render()
        self.height = 400
        self.scale = self.width / (self.env.x_threshold * 2) #pixel/local unit

        self.image_width = int(200)
        self.image_height = int(self.scale * (2 * self.env.length)) #polelen 

        #self.x_threshold = 1
        #self.env.unwrapped.x_threshold = 1 #Works, but overrides also the pole length when rendering

        #---Redefine observation space---#        
        self.observation_space = spaces.Box(low=0., high=1., shape=(self.image_height, self.image_width), dtype=np.float32)

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        #if (observation[0] < -self.x_threshold) or (observation[1] > self.x_threshold):
            #self.env.seed()
        #    done = True  #Terminate if the cartpole leaves the cropped region (not necessary)
        
        return self.observation(observation), reward, done, info

    def observation(self, obs):
        """
        The image is cropped around the cartpole position, meaning that the CNN
        always sees the cartpole as "stationary", with only the pole moving.
        This makes learning much easier, but removes any information about the
        cartpole position, which needs to be accounted by modifying the reward function.
        """
        
        x_cartpole_pixel = int(obs[0] * self.scale + (self.width / 2)) #obs[0]
        y_cartpole_pixel = int(self.height - 100)

        frame = self.env.render(mode='rgb_array')[..., 0]
        #Pick just the first channel: (400, 600, 3) -> (400, 600)

        frame = 1. - (frame / 255) #Normalize in [0,1] and invert colors (black background and white cartpole)
        frame[np.nonzero(frame)] = 1. #All of the same color

        #---Crop the area around the cartpole---#
        return np.resize(frame[y_cartpole_pixel-int(self.image_height):y_cartpole_pixel,
                     x_cartpole_pixel-self.image_width//2:x_cartpole_pixel+self.image_width//2], (125, 200))
