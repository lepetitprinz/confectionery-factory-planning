import common.config as config

import numpy as np
import pandas as pd


class Human(object):
    key_human_capa = config.key_human_capa      # Human capacity
    key_human_usage = config.key_human_usage    # Human usage

    def __init__(self, demand: pd.DataFrame, result: pd.DataFrame, cstr):
        self.demand = demand
        self.result = result
        self.capacity = cstr[self.key_human_capa]
        self.usage = cstr[self.key_human_usage]

    def apply(self):
        pass

    def get_dmd_list_by_floor(self):
        pass

    def get_prod_start_time(self):
        pass

    def determine_prod_dmd(self):
        pass