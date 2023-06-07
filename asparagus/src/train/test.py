import torch
import numpy as np
import os
import sys

from .. import utils
try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy import stats

class TestModel:

    def __init__(self,model,data_test,checkpoint):
        self.model = model
        self.data_test = data_test
        self.checkpoint = checkpoint

        latest_ckpt = utils.file_managment.load_checkpoint(self.checkpoint)
        self.model.load_state_dict(latest_ckpt['model_state_dict'])

    def test(self,plot=False,save_csv=False):
        #TODO: Add code



