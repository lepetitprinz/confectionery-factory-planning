import common.util as util

import pandas as pd
from typing import List, Dict
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeqModel(object):
    time_limit = 30
    make_span = False
    optput_flag = True

    res_type_capa_map = {
        'NOR': 1,
        'PPL': 2
    }

    def __init__(self, cstr_cfg: dict, plant: str, dmd_due: dict, item_res_duration: dict, job_change=None):
        # Constraint attribute
        self.cstr_cfg = cstr_cfg

        # Plant instance attribute
        self.plant = plant

        # Demand instance attribute
        self.dmd_due_date = dmd_due
        self.max_due_date = 0
        self.max_due_day = 0

        # Resource instance attribute
        self.add_res_people_yn = False    # True / False
        self.res_human_map = {}

        # Capacity instance attribute
        self.work_days = 5
        self.sec_of_day = 86400
        self.capa_type = 'daily_capa'
        self.res_start_time_of_day = 0

        # Duration instance attribute
        self.res_default_duration = 60
        self.item_res_duration = item_res_duration

        # Job change instance attribute
        self.job_change = job_change

    def init(self, dmd_list: list, res_grp_list: dict):
        self.set_max_due_date()

        model = Model(name='lotte')

        # Set resource
        model_res = self.set_resource(model=model, res_grp_list=res_grp_list)

        model = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            res_grp_list=res_grp_list,
            model_res=model_res
        )

        model = self.set_parameter(model=model)

        return model

    def set_max_due_date(self):
        due_list = []
        for sku_due in self.dmd_due_date.values():
            for due_date in sku_due.values():
                due_list.append(due_date)

        due_list = sorted(due_list, reverse=True)

        self.max_due_date = due_list[0]
        self.max_due_day = round(due_list[0] / self.sec_of_day)

    @staticmethod
    def optimize(model: Model):
        model.optimize()

    # Set work defined
    def set_activity(self, model: Model, dmd_list: list, res_grp_list: dict, model_res) -> Model:
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])
            activity[act_name] = model.addActivity(
                name=f'Act[{act_name}]',
                duedate=due_date,
                weight=1    # Penalty per unit time when the work completion time is rate for delivery
            )

            # Set mode
            activity[act_name] = self.set_mode(
                act=activity[act_name],
                dmd_id=dmd_id,
                item_cd=item_cd,
                qty=qty,
                res_list=res_grp_list[res_grp_cd],
                model_res=model_res[res_grp_cd],
                res_grp_cd=res_grp_cd
            )

        return model

    # Set work processing method
    def set_mode(self, act: Activity, dmd_id: str, item_cd: str, qty: int, res_list: list,
                 model_res, res_grp_cd: str) -> Activity:
        for res_cd, capacity, unit_cd, res_type in res_list:
            if res_type == 'NOR':    # Check if resource is machine
                # Make each mode (set each available resource)

                duration = self.calc_duration(item_cd=item_cd, res_cd=res_cd, qty=qty)

                mode = Mode(
                    name=f'Mode[{dmd_id}@{res_cd}]',
                    duration=duration    # Duration : the working time of the mode
                )

                # Add break for each mode
                # mode.addBreak(start=0, finish=0)

                # Add resource for each mode(resource)
                mode = self.add_resource(
                    mode=mode,
                    resource=model_res[res_cd],
                    duration=duration
                )

                # add mode list to activity
                act.addModes(mode)

        return act

    def calc_duration(self, item_cd, res_cd, qty) -> int:
        # Calculate duration
        item_res_duration = self.item_res_duration.get(item_cd, None)

        if item_res_duration is None:
            duration_per_unit = self.res_default_duration
        else:
            duration_per_unit = item_res_duration.get(res_cd, self.res_default_duration)

        duration = int(qty * duration_per_unit)

        return duration

    def set_resource(self, model: Model, res_grp_list) -> Dict[str, Dict[str, Resource]]:
        model_res_grp = {}
        for res_grp, res_list in res_grp_list.items():
            model_res = {}
            for res, capacity, unit_cd, res_type in res_list:
                # Add resource object
                add_res = model.addResource(name=res, capacity={(0, "inf"): 1})

                # Add capacity of each resource
                if self.cstr_cfg['apply_res_capacity']:
                    add_res = self.set_capacity(res=add_res, capacity=capacity, unit_cd=unit_cd)

                # map resource code to model resource object
                model_res[res] = add_res
            model_res_grp[res_grp] = model_res

        return model_res_grp

    def set_capacity(self, res: Resource, capacity, unit_cd) -> Resource:
        if self.capa_type == 'daily_capa':
            capa = 0
            if unit_cd == 'MIN':
                capa = capacity * 60 // self.work_days
            elif unit_cd == 'SEC':
                capa = capacity // self.work_days

            for i in range(self.max_due_day + 1):
                start_time = i * self.sec_of_day
                res.addCapacity(start_time, start_time + capa, 1)

        return res

    # Add the specified resource which amount required when executing the mode
    def add_resource(self, mode: Mode, resource, duration: int) -> Mode:
        # Add resource
        mode.addResource(    # ToDo: need to revise requirement
            resource=resource,
            requirement={(0, duration): 1},    # requirement : gives the required amount of resources
            # requirement={(0, 1): 1}  # requirement : gives the required amount of resources
        )

        return mode

    def set_parameter(self, model: Model) -> Model:
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span
        params.OutputFlag = self.optput_flag

        model.Params = params

        return model
