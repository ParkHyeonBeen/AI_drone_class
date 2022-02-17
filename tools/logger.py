import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time



from WSs.WS_FIP.fip_config import FIP_config

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

config = FIP_config()
root = config.path + "results"

class logger():
    def __init__(self, new_data = "none"):

        self.log_dir = ""
        self.data_dir = ""

        self.new_data = new_data
        self.newData = {"name" : new_data,
                    "path" : "",
                    "data" : []}

        self.states = {"name" : "states",
                    "path" : "",
                    "data" : np.array([])}

        self.reward = {"name" : "reward",
                        "path" : "",
                        "data" : []}

    def create_log_directories(self):

        if not os.path.isdir(root): os.mkdir(root)
        self.log_dir = root + "/%s" % time.strftime("%Y%m%d-%H%M%S")
        self.policy_dir = self.log_dir + "/policy/"

        self.data_dir = self.log_dir + "/data_dir/"

        if not os.path.isdir(self.log_dir): os.mkdir(self.log_dir)
        if not os.path.isdir(self.policy_dir): os.mkdir(self.policy_dir)
        if not os.path.isdir(self.data_dir): os.mkdir(self.data_dir)

    def get_data(self, Data, newData = float('nan')):

        states, reward = Data

        if self.states["data"].size == 0:
            self.states["data"] = states
        else:
            self.states["data"] = np.vstack((self.states["data"], states))

        self.reward["data"].append(reward)

        if not math.isnan(newData):
            self.newData["data"].append(newData)

    def log_csv(self):

        newData = np.array(self.newData["data"])
        reward = np.array(self.reward["data"])
        newData = newData.reshape(-1, 1)
        reward = reward.reshape(-1, 1)

        if self.new_data != "none":
            df = np.hstack((self.states["data"], reward, newData))
            df = pd.DataFrame(df)

        else:
            df = np.hstack((self.states["data"], reward))
            df = pd.DataFrame(df)

        df.to_csv(self.data_dir + "log_data.csv", index=False)

    def log_policy(self, pi):
        torch.save(pi.state_dict(), self.policy_dir + "policy")

    def state_trajectory(self): return self.states["data"]


