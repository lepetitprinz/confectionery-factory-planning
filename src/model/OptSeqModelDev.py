import common.util as util
import common.config as config

from itertools import permutations
from typing import List, Dict
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeqModel(object):
    time_limit = 30
    make_span = False
    optput_flag = True

    col_res_grp = config.col_res_grp

    def __init__(self, cstr_cfg: dict, plant: str, plant_data: dict):
        # Constraint attribute
        self.cstr_cfg = cstr_cfg

        # Plant instance attribute
        self.plant = plant

        # Demand instance attribute
        self.dmd_due_date = plant_data['demand']['plant_dmd_due'][plant]
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
        self.item_res_duration = plant_data['resource']['plant_item_res_duration'][plant]

        # Job change instance attribute
        self.sku_to_item = plant_data['sku_to_brand']
        self.job_change = plant_data['job_change'].get(plant, None)

    def init(self, dmd_list: list, res_grp_list: dict):
        # Step0. Set the due date
        self.set_max_due_date()

        # Step1. Instantiate the model
        model = Model(name='lotte')

        # Step2. Set the resource
        model_res = self.set_resource(model=model, res_grp_list=res_grp_list)

        # Step3. Set the activity
        model, activity = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            res_grp_list=res_grp_list,
            model_res=model_res
        )

        # Step4. Set the job change activity (Optional)
        if self.cstr_cfg['apply_job_change']:
            model = self.set_job_change_activity(
                model=model,
                activity=activity,
                dmd_list=dmd_list,
                res_grp_list=res_grp_list,
                model_res=model_res
            )

        model = self.set_model_parameter(model=model)

        return model

    def set_job_change_available_res_grp(self, dmd_list: list):
        job_change_res_grp = list(self.job_change)
        res_grp_dmd = {}
        job_change_act_list = []
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            if res_grp_cd in job_change_res_grp:
                act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])
                job_change_act_list.append(act_name)
                if res_grp_cd not in res_grp_dmd:
                    res_grp_dmd[res_grp_cd] = [act_name]
                else:
                    res_grp_dmd[res_grp_cd].append(act_name)

        return res_grp_dmd, job_change_act_list

    def set_job_change_activity(self, model: Model, activity: dict, dmd_list, res_grp_list: dict, model_res: dict):
        # Define state
        state = model.addState(name='Job_Change')
        state.addValue(time=0, value=0)

        res_grp_dmd, job_change_act_list = self.set_job_change_available_res_grp(dmd_list=dmd_list)
        # Set state
        act_state = self.set_state(data=job_change_act_list)

        for res_grp_cd, act_list in res_grp_dmd.items():
            res_mode = self.set_job_change_mode(
                act_list=act_list,
                res_grp_cd=res_grp_cd,
                state=state,
                act_state=act_state,
                res_cd_list=res_grp_list[res_grp_cd],
                res_model=model_res[res_grp_cd],
            )
            job_change_activity = self.add_job_change_activity(
                model=model,
                mode=res_mode,
                act_list=act_list,
            )
            model = self.add_job_change_temporal(model, activity, job_change_activity)

        return model

    def add_job_change_temporal(self, model, activity, job_change_activity):
        for job_change_act in job_change_activity:
            model.addTemporal(job_change_activity[job_change_act], activity[job_change_act], 'CS')
            model.addTemporal(activity[job_change_act], job_change_activity[job_change_act], 'SC')

        return model

    def add_job_change_activity(self, model: Model, mode: dict, act_list: list):
        job_change_activity = {}
        for act in act_list:
            job_change_activity[act] = model.addActivity(name=f'Setup[{act}]', autoselect=True)
            for from_act, to_act in mode:
                if act == to_act:
                    job_change_activity[act].addModes(mode[(from_act, to_act)])

        return job_change_activity

    def set_job_change_mode(self, act_list, res_grp_cd, state, act_state, res_cd_list, res_model):
        act_seq_list = list(permutations(act_list, 2))
        res_mode = {}
        for from_act, to_act in act_seq_list:
            # Get job change time
            job_change_time = self.get_job_change_time(res_grp_cd=res_grp_cd, from_act=from_act, to_act=to_act)
            if job_change_time != 0:
                for res_cd in res_cd_list:
                    res_mode[(from_act, to_act)] = Mode(
                        name=f'Mode_{from_act}_{to_act}_{res_cd[0]}',
                        duration=job_change_time
                    )
                    res_mode[(from_act, to_act)].addState(
                        state=state,
                        fromValue=act_state[from_act],
                        toValue=act_state[to_act]
                    )
                    res_mode[(from_act, to_act)].addResource(
                        resource=res_model[res_cd[0]],
                        requirement={(0, job_change_time): 1}
                    )

        return res_mode

    def get_job_change_time(self, res_grp_cd, from_act, to_act):
        from_res_cd = self.sku_to_item[from_act.split('@')[1]]  # From brand
        to_res_cd = self.sku_to_item[to_act.split('@')[1]]  # To brand
        job_change_time = self.job_change[res_grp_cd].get((from_res_cd, to_res_cd), 0)

        return job_change_time

    def set_state(self, data: list):
        state = {act: i for i, act in enumerate(data)}

        return state

    def set_max_due_date(self):
        due_list = []
        for sku_due in self.dmd_due_date.values():
            for due_date in sku_due.values():
                due_list.append(due_date)

        due_list = sorted(due_list, reverse=True)

        self.max_due_date = due_list[0]
        self.max_due_day = round(due_list[0] / self.sec_of_day)

    def set_activity(self, model: Model, dmd_list: list, res_grp_list: dict, model_res: dict) -> Model:
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            # Make the activity naming
            act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])

            # Define activities
            activity[act_name] = model.addActivity(
                name=f'Act[{act_name}]',
                duedate=due_date,
                weight=1,    # Penalty per unit time when the work completion time is rate for delivery
            )

            # Set modes
            activity[act_name] = self.set_mode(
                act=activity[act_name],
                dmd_id=dmd_id,
                item_cd=item_cd,
                qty=qty,
                res_list=res_grp_list[res_grp_cd],
                model_res=model_res[res_grp_cd]
            )

        return model, activity

    # Set work processing method
    def set_mode(self, act: Activity, dmd_id: str, item_cd: str, qty: int, res_list: list, model_res: dict) -> Activity:
        for res_cd, capacity, unit_cd, res_type in res_list:
            # Calculate the duration (the working time of the mode)
            duration = self.calc_duration(item_cd=item_cd, res_cd=res_cd, qty=qty)

            # Make each mode (set each available resource)
            mode = Mode(name=f'Mode[{dmd_id}@{res_cd}]', duration=duration)

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

    # Add the specified resource which amount required when executing the mode
    def add_resource(self, mode: Mode, resource, duration: int) -> Mode:
        # requirement : gives the required amount of resources
        mode.addResource(resource=resource, requirement={(0, duration): 1},)

        return mode

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

    def set_model_parameter(self, model: Model) -> Model:
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span
        params.OutputFlag = self.optput_flag

        model.Params = params

        return model

    @staticmethod
    def optimize(model: Model):
        model.optimize()