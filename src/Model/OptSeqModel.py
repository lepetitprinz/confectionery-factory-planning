import common.util as util

import pandas as pd
import numpy as np
from optimize.optseq import Model, Mode, Parameters


class OptSeqModel(object):
    time_limit = 100
    make_span = False

    def __init__(self, item_res_grp, res_grp, res_grp_item, res_grp_duration: dict):
        self.item_res_grp = item_res_grp    # item -> available resource group list

        # Resource configuration
        self.res_grp = res_grp
        self.res_grp_item = res_grp_item
        self.res_grp_duration = res_grp_duration   # Resource group -> Making time

    def init(self, dmd_list: list):
        model = Model(name='lotte')

        resource = self.set_resource()

        model = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            resource=resource
        )

        model = self.set_parameter(model=model)

        return model

    def set_activity(self, model: Model, dmd_list: list, resource):
        activity = {}
        for dmd_id, item_cd, qty, due_date in dmd_list:
            act_name = util.generate_name(dmd_id, item_cd)
            activity[act_name] = model.addActivity(name=f'Act[{act_name}]', duedate=due_date)

            # Set mode
            mode_list = self.set_mode(item_cd=item_cd, resource=resource)

            # add mode list to activity
            activity[act_name].addModes(mode_list)

        return model

    def set_mode(self, item_cd, resource) -> list:
        mode_list = []
        for res_grp in self.item_res_grp[item_cd]:
            # Make each mode (set each available resource)
            mode = Mode(f'Mode[{res_grp}]', duration=self.res_grp_duration[res_grp])

            # Add resource for each mode(resource)
            self.add_resource(mode=mode, res_grp=res_grp, resource=resource)

        return mode_list

    def set_resource(self):
        pass

    def set_capacity(self):
        pass

    def add_resource(self, mode: Mode, res_grp: str, resource):

        return mode

    def set_break(self):
        pass

    def set_parameter(self, model: Model):
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span

        model.Params = params

        return model