import utils.setup_path
import airsim
import numpy as np
import math
import time

import gym
from gym import spaces
from airgym.envs.airsim_env import AirSimEnv

import numpy as np
from numpy import savetxt

"""
X=0,Y=0,Z=0,Yaw=0
"""
track0 = [
                (0, -1), (128, -1), (128, -128), (0, -128),
                (0, -1),
            ]
bound0 = [
                (-2, 1), (130, 1), (130, -130), (-2, -130),
                (-2, 1),
            ]
            
"""
X=0,Y=0,Z=0,Yaw=180
"""
track1 = [
                (0, -1), (-127, -1), (-127, -128), (0, -128),
                (0, -1),
            ]
bound1 = [
                (2, 1), (-129, 1), (-129, -130), (2, -130),
                (2, 1),
            ]
"""
X=0,Y=0,Z=0,Yaw=0
"""
track2 = [
                (0, -1), (-128, -1), (-128, -128), (0, -128),
                (0, -1),
            ]
bound2 = [
                (-2, 1), (130, 1), (130, -130), (-2, -130),
                (-2, 1),
            ]

#action_file = "action.csv"
class AirSimCarEnvContAction(AirSimEnv):
    def __init__(self, ip_address, image_shape):
        super().__init__(image_shape)

        self.image_shape = image_shape
        self.start_ts = 0

        self.state = {
            "position": np.zeros(3),
            "prev_position": np.zeros(3),
            "pose": None,
            "prev_pose": None,
            "collision": False,
            "orientation": None
        }

        self.car = airsim.CarClient(ip=ip_address)
        self.car.confirmConnection()
        self.car.enableApiControl(True) #這裡是多加的 為了能用API
        
        high = np.array([0.9, 1]) #方向，煞車油門
        low = np.array([-0.9, 0])
        
        self.action_space = spaces.Box(low , high)

        #self.image_request = airsim.ImageRequest(
        #    "0", airsim.ImageType.DepthPerspective, True, False
        #)
        self.image_request = airsim.ImageRequest(
            "0", airsim.ImageType.Scene, False, False
        )
        #CameraID, Data type, pixels as float or not, compressed or not

        self.car_controls = airsim.CarControls()
        self.car_state = None
        
        self.static_count = 0
        
        self.track = [(track0, bound0), (track1, bound1), (track2, bound2)]
        
        DamnAnimals = [] #Forgive me for cursing the animals which always breaks my training
        print("Animals: ")
        for objects in self.car.simListSceneObjects():
            if "Raccoon" in objects or "Deer" in objects:
                print(objects)
                DamnAnimals.append(objects)
                
        for animals in DamnAnimals:
            self.car.simDestroyObject(animals)
            time.sleep(0.05)
        print("Animals Cleaned Over.")

    def _setup_car(self):
        self.car.reset()
        self.car.enableApiControl(True)
        self.car.armDisarm(True)
        time.sleep(0.01)

    def __del__(self):
        self.car.reset()

    def _do_action(self, action):
        if float(action[1]) > 0.5 :
            self.car_controls.throttle = 1
            self.car_controls.brake = 0
        else:
            self.car_controls.throttle = 0
            self.car_controls.brake = 1
        
        print("action = ", action)
        """
        with open(action_file, 'a') as f:
            savetxt(f, action, delimiter = ',')
        """
        #print("action type = ", type(action))
        self.car_controls.steering = float(action[0])
        
        
        
        self.car.setCarControls(self.car_controls)
        time.sleep(0.8)

    def transform_obs(self, response):
        #img1d = np.array(response.image_data_float, dtype=np.float)
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #Using Scene
        #img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        #img2d = np.reshape(img1d, (response.height, response.width))
        img2d = img1d.reshape(response.height, response.width, 3)
        # original image is fliped vertically
        #img2d = np.flipud(img2d)


        from PIL import Image

        print(type(img2d))
        print("img2d size", img2d.shape)
        image = Image.fromarray(img2d)
        im_final = np.array(image.resize((84, 84)).convert("L"))

        return im_final.reshape([84, 84, 1])

    def _get_obs(self):
        responses = self.car.simGetImages([self.image_request])
        image = self.transform_obs(responses[0])

        self.car_state = self.car.getCarState()

        self.state["prev_pose"] = self.state["pose"]
        self.state["pose"] = self.car_state.kinematics_estimated
        self.state["collision"] = self.car.simGetCollisionInfo().has_collided
        self.state["orientation"] = self.car_state.kinematics_estimated.orientation

        return image
        
    def Quaternion_Z_deg(self, x): #x must be Quaternionr in airsim.type
        r = 2*math.acos(x.w_val) #in rad
        return r*180/math.pi #in deg
        
    def mid_line_dist(self):
        pts = [
            np.array([x, y, 0])
            for x, y in self.track[self.track_num][0]
        ]
        car_pt = self.state["pose"].position.to_numpy_array()

        dist = 10000000
        for i in range(0, len(pts) - 1):
            #wheather the car_pt projection is between the pts or not
            segment = np.linalg.norm(pts[i] - pts[i + 1])
            project1 = np.linalg.norm(np.dot(car_pt - pts[i], (pts[i + 1] - pts[i]))) / segment
            project2 = np.linalg.norm(np.dot(car_pt - pts[i + 1], (pts[i] - pts[i + 1]))) /  segment
            if project1 > segment :
                dist = min(
                    dist,
                    np.linalg.norm(car_pt - pts[i + 1]))
            elif project2 > segment:
                dist = min(
                    dist,
                    np.linalg.norm(car_pt - pts[i]))
            else:
                dist = min(
                    dist,
                    np.linalg.norm(
                        np.cross((car_pt - pts[i]), (car_pt - pts[i + 1]))
                    )
                    / segment,
                )
        return dist
        
    def bound_dist(self):
        bound_track = self.track[self.track_num][1]
        bound = [
            np.array([x, y, 0])
            for x, y in bound_track
        ]
        car_pt = self.state["pose"].position.to_numpy_array()
        bound_dist_sum = 0
        for i in range(0, len(bound) - 1):
            bound_dist_sum += np.linalg.norm(np.cross((car_pt - bound[i]), (car_pt - bound[i+1])) / np.linalg.norm(bound[i] - bound[i+1]))
            
        bound_dist_sum -= (np.linalg.norm(bound[0] - bound[1]) + np.linalg.norm(bound[1] - bound[2]))
        bound_dist_sum /= 2
            
        return bound_dist_sum
        

    def _compute_reward(self):
        MAX_SPEED = 10 #原先似乎太大   <-------------------
        MIN_SPEED = 0   #<----------------------

        THRESH_DIST = 3.5
        BETA = 1  #<-------------------------
        dist = self.mid_line_dist()
        bound_dist_sum = self.bound_dist()
        done = 0            
        if dist > THRESH_DIST:
            reward = -1
            done = 1
            print("Done -- distance Out\n")
        else:
            reward_dist = math.exp(-((dist * 0.84 / BETA)**2))  #<-------------------------*0.84 因為exp(-0.84^2)=0.5，BETA就能代表reward=0的距離
            
            speed = self.car_state.speed
            if speed < 3:
                self.static_count += 1
            else:
                self.static_count = 0
            
            if speed > MAX_SPEED:
                speed = MAX_SPEED
            reward_speed = (
                (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            )
            
            #reward_deg = abs(Quaternion_Z_deg(self.state["orientation"]))
            reward_bound = - (bound_dist_sum**2)
            
            #reward = reward_dist * reward_speed + reward_bound
            reward = reward_dist * reward_speed

            print("position = ",self.state["pose"].position.to_numpy_array())
            print("%-10s" % "dist rew",': %8.3f'%reward_dist, "%-6s" % "dist", ': %.3f'%dist)
            print("%-10s" % "speed rew", ': %8.3f'%reward_speed, "%-6s" % "speed", ': %.3f'%self.car_state.speed)
            print("%-10s" % "bound rew", ': %8.3f'%reward_bound, "%-6s" % "bound", ': %.3f'%bound_dist_sum)
            print("%-10s" % "reward", ': %8.3f'%float(reward))
            print()
            
            if self.state["collision"]:
                reward = -1
                print("Done -- collision\n")
                done = 1
            elif self.static_count > 5:
                reward = -1
                done = 1
                print("Done -- Static\n")
            
        #reset static count
        if done == 1:
            self.static_count = 0


        return reward, done

    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state

    def reset(self):
        self._setup_car()
        self._do_action([0, 0.5])
        return self._get_obs()
    
    def setkwargs(self, **kwargs):
        self.track_num = kwargs['track']
