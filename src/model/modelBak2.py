import common.util as util
import common.config as config

from itertools import permutations
from typing import Dict, Tuple, List
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeqModel(object):
    # Data dictionary key configuration
    key_jc = config.key_jc
    key_sku_type = config.key_sku_type

    # Model parameter option
    time_limit = config.time_limit
    make_span = config.make_span
    optput_flag = config.optput_flag
    max_iteration = config.max_iteration
    report_interval = config.report_interval
    back_truck = config.back_truck

    col_res_grp = config.col_res_grp

    def __init__(self, exec_cfg: dict, cstr_cfg: dict, except_cfg: dict, plant: str, plant_data: dict):
        # Constraint attribute
        self.exec_cfg = exec_cfg
        self.cstr_cfg = cstr_cfg
        self.except_cfg = except_cfg

        # Capacity instance attribute
        self.work_days = 5
        self.schedule_weeks = 4
        self.sec_of_day = 86400
        self.plant_start_hour = 25200    # 25200(sec) = 7(hour) * 60 * 60
        self.res_avail_time = plant_data['resource']['plant_res_avail_time'][plant]

        # Duration instance attribute
        self.time_unit = 'M'
        self.res_default_duration = 1
        self.item_res_duration = plant_data['resource']['plant_item_res_duration'][plant]

        # Job change instance attribute
        self.start_act = 'start@start'
        self.sku_to_type = plant_data[self.key_sku_type]
        self.job_change = plant_data[self.key_jc].get(plant, None)

    def init(self, dmd_list: list, res_grp_dict: dict):
        # Step1. Instantiate the model
        model = Model(name='lotte')

        # Step2. Set the resource
        model_res, res_grp_dict = self.set_resource(model=model, res_grp_dict=res_grp_dict)

        # Step3. Set the activity
        model, activity, rm_act_list = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            res_grp_dict=res_grp_dict,
            model_res=model_res
        )

        # Step4. Set the job change activity (Optional)
        if self.cstr_cfg['apply_job_change']:
            # Filter demand
            dmd_list = self.filter_demand(demand=dmd_list, rm_act=rm_act_list)

            model = self.set_job_change_activity(
                model=model,
                activity=activity,
                dmd_list=dmd_list,
                res_grp_list=res_grp_dict,
                model_res=model_res
            )

        model = self.set_model_parameter(model=model)

        return model, rm_act_list

    # Set resource
    def set_resource(self, model: Model, res_grp_dict) -> Tuple[Dict[str, Dict[str, Resource]], dict]:
        model_res_grp = {}
        res_grp_list_copy = {**res_grp_dict}

        for res_grp, res_list in res_grp_list_copy.items():
            model_res = {}
            for resource in res_list:
                # Add available time of each resource
                add_res = model.addResource(name=resource)

                if self.cstr_cfg['apply_res_available_time']:
                    avail_time = self.res_avail_time.get(resource, None)
                    if avail_time:
                        add_res = self.add_res_capacity(res=add_res, avail_time=avail_time)
                    else:
                        # Remove resource candidate from resource group
                        res_grp_dict[res_grp].remove(resource)
                        continue

                else:
                    # Add infinite capacity resource
                    add_res = model.addResource(name=resource, capacity={(0, "inf"): 1})

                model_res[resource] = add_res

            if len(model_res) != 0:
                model_res_grp[res_grp] = model_res
            else:
                res_grp_dict.pop(res_grp)

        return model_res_grp, res_grp_dict

    def add_res_capacity(self, res: Resource, avail_time) -> Resource:
        time_multiple = 1
        if self.time_unit == 'M':
            time_multiple = 60

        start_time = self.plant_start_hour
        for i, time in enumerate(avail_time * self.schedule_weeks):
            if not isinstance(time, int):
                raise TypeError(f"Time is not integer - resource: {res.name}")

            end_time = min(start_time + time * time_multiple, start_time + self.sec_of_day - self.plant_start_hour)

            # Add the capacity
            res.addCapacity(start_time, end_time, 1)
            start_time += self.sec_of_day

            if (i+1) % 5 == 0:    # skip saturday & sunday
                start_time += self.sec_of_day * 2

        # Exception for over demand
        res.addCapacity(start_time + self.sec_of_day, 'inf', 1)

        return res

    # Set activity
    def set_activity(self, model: Model, dmd_list: list, res_grp_dict: dict, model_res: dict) \
            -> Tuple[Model, dict, List[str]]:
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            if res_grp_dict.get(res_grp_cd, None):   # Todo : Filter demand that does not exist in resource group info
                # Make the activity naming
                act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])

                # Define activities
                activity[act_name] = model.addActivity(
                    name=f'Act[{act_name}]',
                    duedate=due_date,
                    # duedate='inf',
                    weight=1,    # Penalty per unit time when the work completion time is rate for delivery
                )

                # Set modes
                act = self.set_mode(
                    act=activity[act_name],
                    dmd_id=dmd_id,
                    item_cd=item_cd,
                    qty=qty,
                    res_list=res_grp_dict[res_grp_cd],
                    model_res=model_res[res_grp_cd]
                )
                activity = self.remove_empty_mode_from_act(activity=activity, act_name=act_name, act=act)

        model, rm_act_list = self.remove_empty_mode_from_model(model=model)

        return model, activity, rm_act_list

    # Set work processing method
    def set_mode(self, act: Activity, dmd_id: str, item_cd: str, qty: int, res_list: list, model_res: dict):
        for resource in res_list:
            # Calculate the duration (the working time of the mode)
            duration_per_unit = self.get_duration_per_unit(item_cd=item_cd, res_cd=resource)

            if duration_per_unit is not None:
                duration = int(qty * duration_per_unit)

                if duration <= 0:
                    raise ValueError(f"Duration is not positive integer: item: {item_cd} resource: {resource}")

                # Make each mode (set each available resource)
                mode = Mode(name=f'Mode[{dmd_id}@{item_cd}@{resource}]', duration=duration)

                # Add break for each mode
                mode.addBreak(start=0, finish=duration, maxtime='inf')

                # Add resource for each mode(resource)
                mode = self.add_resource(
                    mode=mode,
                    resource=model_res[resource],
                    duration=duration
                )

                # add mode list to activity
                act.addModes(mode)

        return act

    @staticmethod
    # Add the specified resource which amount required when executing the mode
    def add_resource(mode: Mode, resource, duration: int) -> Mode:
        # requirement : gives the required amount of resources
        mode.addResource(resource=resource, requirement={(0, duration): 1}, rtype=None)

        return mode

    def get_duration_per_unit(self, item_cd, res_cd) -> int:
        duration_per_unit = None

        # Calculate duration
        item_res_duration = self.item_res_duration.get(item_cd, None)

        if item_res_duration is None:
            if self.except_cfg['miss_duration'] == 'add':
                duration_per_unit = self.res_default_duration

            if self.exec_cfg['verbose']:
                print(f"Item: {item_cd} does not have any resource duration.")
        else:
            duration_per_unit = item_res_duration.get(res_cd, None)
            if (duration_per_unit is None) or (duration_per_unit == 0):
                if self.except_cfg['miss_duration'] == 'add':
                    duration_per_unit = self.res_default_duration
                else:
                    duration_per_unit = None

                if self.exec_cfg['verbose']:
                    print(f"Item: {item_cd} - {res_cd} does not have duration")

        return duration_per_unit

    def set_model_parameter(self, model: Model) -> Model:
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span
        params.OutputFlag = self.optput_flag
        params.MaxIteration = self.max_iteration
        params.Backtruck = self.back_truck
        # params.ReportInterval = self.report_interval

        model.Params = params

        return model

    @staticmethod
    def remove_empty_mode_from_act(activity: dict, act_name: str, act: Activity):
        if len(act.modes) > 0:
            activity[act_name] = act
        else:
            activity.pop(act_name)

        return activity

    @staticmethod
    def remove_empty_mode_from_model(model: Model):
        rm_act_list = []
        for act in model.act[:]:
            if len(act.modes) == 0:
                rm_act_list.append(act.name)
                model.act.remove(act)

        return model, rm_act_list

    @staticmethod
    def filter_demand(demand, rm_act):
        rm_dmd_list = [act[act.index('[') + 1: act.index('@')] for act in rm_act]
        demand_filtered = [dmd for dmd in demand if dmd[0] not in rm_dmd_list]

        return demand_filtered

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

    def set_job_change_available_res_grp(self, dmd_list: list):
        job_change_avail_res_grp = list(self.job_change)
        res_grp_dmd = {}
        job_change_act_list = []
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            if res_grp_cd in job_change_avail_res_grp:
                act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])
                job_change_act_list.append(act_name)
                if res_grp_cd not in res_grp_dmd:
                    res_grp_dmd[res_grp_cd] = [act_name]
                else:
                    res_grp_dmd[res_grp_cd].append(act_name)

        return res_grp_dmd, job_change_act_list

    @staticmethod
    def add_job_change_temporal(model, activity, job_change_activity):
        for job_change_act in job_change_activity:
            model.addTemporal(job_change_activity[job_change_act], activity[job_change_act], 'CS')
            model.addTemporal(activity[job_change_act], job_change_activity[job_change_act], 'SC')

        return model

    def add_job_change_activity(self, model: Model, mode: dict, act_list: list):
        job_change_activity = {}
        for act in act_list:
            if act != self.start_act:
                job_change_activity[act] = model.addActivity(name=f'Setup[{act}]', autoselect=True)
                for from_act, to_act in mode:
                    if act == to_act:
                        job_change_activity[act].addModes(mode[(from_act, to_act)])

        return job_change_activity

    def set_job_change_mode(self, act_list, res_grp_cd, state, act_state, res_cd_list, res_model):
        act_list.append(self.start_act)

        act_seq_list = list(permutations(act_list, 2))
        res_mode = {}
        for from_act, to_act in act_seq_list:
            # Get job change time
            job_change_time = self.get_job_change_time(res_grp_cd=res_grp_cd, from_act=from_act, to_act=to_act)
            for res_cd in res_cd_list:
                res_mode[(from_act, to_act)] = Mode(
                    name=f'Mode_setup[{from_act}|{to_act}|{res_cd[0]}]',
                    duration=job_change_time
                )
                #
                # res_mode[(from_act, to_act)].addBreak(start=0, finish=job_change_time, maxtime='inf')

                res_mode[(from_act, to_act)].addState(
                    state=state,
                    fromValue=act_state[from_act],
                    toValue=act_state[to_act]
                )
                if job_change_time != 0:
                    res_mode[(from_act, to_act)].addResource(
                        resource=res_model[res_cd[0]],
                        requirement={(0, job_change_time): 1}
                    )

        return res_mode

    def get_job_change_time(self, res_grp_cd, from_act, to_act) -> int:
        from_res_cd = self.sku_to_type.get(from_act.split('@')[1], None)  # From brand
        to_res_cd = self.sku_to_type.get(to_act.split('@')[1], None)  # To brand
        job_change_time = int(self.job_change[res_grp_cd].get((from_res_cd, to_res_cd), 0))

        return job_change_time

    def set_state(self, data: list):
        state = {act: (i + 1) for i, act in enumerate(data)}
        state[self.start_act] = 0

        return state

    def check_model_init_set(self, model: Model):
        # Check the activity
        for act in model.act:
            if len(act.modes) == 0:
                raise ValueError(f"Activity: {act.name} does not have modes.")

        # Check the mode
        if not self.cstr_cfg['apply_job_change']:
            for mode in model.modes:
                if len(mode.requirement) == 0:
                    raise ValueError(f"Mode: {mode.name} does not have resources.")

        # Check the resource
        res_list = set()
        for resource in model.res[:]:
            if resource.capacity == {}:
                model.res.remove(resource)
            res_list.add(resource.name)
            for (from_time, to_time), req in resource.capacity.items():
                if isinstance(from_time, int) + isinstance(from_time, int) + isinstance(from_time, int) != 3:
                    raise TypeError(f"Resource: {resource.name} contains non-int type.")

        # Compare resource & resource in the mode
        act_mode_res_list = set()
        for act in model.act:
            for mode in act.modes:
                if mode.duration != 0:
                    res = list(mode.requirement.keys())[0][0]
                    act_mode_res_list.add(res)

        if len(act_mode_res_list - res_list) > 0:
            raise ValueError(f"Infeasible Setting")

        elif len(res_list - act_mode_res_list) > 0:
            res_filter_list = list(res_list - act_mode_res_list)

            for resource in model.res[:]:
                if resource.name in res_filter_list:
                    model.res.remove(resource)

        return model

    @staticmethod
    def make_act_mode_map(model: Model):
        act_mode_map = {}
        for act in model.act:
            if len(act.modes) == 1:
                act_mode_map[act.name] = act.modes[0].name

        return act_mode_map

    @staticmethod
    def optimize(model: Model):
        model.optimize()
