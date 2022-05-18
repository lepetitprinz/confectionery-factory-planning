import common.util as util
import common.config as config
from common.name import Key

import os
from itertools import permutations
from typing import Dict, Tuple, List
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeqModel(object):
    # job change
    job_change_type = ['BRAND_CHANGE', 'FLAVOR_CHANGE', 'STANDARD_CHANGE']

    # Model parameter option
    time_limit = config.time_limit
    make_span = config.make_span
    optput_flag = config.optput_flag
    max_iteration = config.max_iteration

    def __init__(
            self,
            cfg: dict,
            plant: str,
            plant_data: dict,
            version,
    ):
        # Execution instance attribute
        self.exec_cfg = cfg['exec']
        self.cstr_cfg = cfg['cstr']
        self.except_cfg = cfg['except']

        self.plant = plant
        self.version = version
        self.fp_version = version.fp_version
        self.fp_name = version.fp_version + '_' + version.fp_seq + '_' + plant

        # Name instance attribute
        self.key = Key()

        # Resource instance attribute
        self.res_to_res_grp = {}
        self.res_grp = plant_data[self.key.res][self.key.res_grp][plant]

        # Resource capacity instance attribute
        self.work_days = 5
        self.schedule_weeks = 4
        self.sec_of_day = 86400
        self.plant_start_hour = 0    # 25200(sec) = 7(hour) * 60 * 60

        # Resource Duration instance attribute
        self.time_unit = 'M'
        self.default_res_duration = 1
        self.item_res_duration = plant_data[self.key.res][self.key.res_duration][plant]

        # Path instance attribute
        self.save_path = os.path.join('..', '..', 'result')
        self.optseq_output_path = os.path.join('..', 'operation', 'optseq_output.txt')

        # Constraint
        # Resource available time instance attribute
        if self.cstr_cfg['apply_res_available_time']:
            self.res_capa_days = plant_data[self.key.cstr][self.key.res_avail_time].get(plant, None)

        # Job change instance attribute
        if self.cstr_cfg['apply_job_change']:
            self.job_change = plant_data[self.key.cstr][self.key.jc].get(plant, None)
            self.sku_to_type = plant_data[self.key.cstr][self.key.sku_type]

        # Simultaneous production
        if self.cstr_cfg['apply_sim_prod_cstr']:
            self.sim_prod_cstr = plant_data[self.key.cstr][self.key.sim_prod_cstr].get(plant, None)

    def init(self, plant: str, dmd_list: list, res_grp_dict: dict):
        # Step1. Instantiate the model
        model = Model(name=plant)

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
            self.set_res_to_res_grp()

            model = self.set_job_change_activity(
                model=model,
                activity=activity,
                model_res=model_res
            )

        model = self.set_model_parameter(model=model)

        return model, rm_act_list

    @staticmethod
    def get_res_to_dmd_list(model: Model):
        res_dmd_list = {}
        for activity in model.act:
            for mode in activity.modes:
                resource = mode.name.split('@')[2][:-1]
                act_name = activity.name
                act_name = act_name[act_name.index('[') + 1: act_name.index(']')]
                if resource not in res_dmd_list:
                    res_dmd_list[resource] = [act_name]
                else:
                    res_dmd_list[resource].append(act_name)

        return res_dmd_list

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
                    capa_days = self.res_capa_days.get(resource, None)
                    if capa_days:
                        add_res = self.add_res_capacity(res=add_res, capa_days=capa_days)
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

        if self.cstr_cfg['apply_sim_prod_cstr']:
            if self.sim_prod_cstr is not None:
                model_res_grp = self.add_virtual_resource(
                    model=model,
                    model_res_grp=model_res_grp,
                )

        return model_res_grp, res_grp_dict

    def add_virtual_resource(self, model: Model, model_res_grp: dict,):
        for res_grp, res_map in self.sim_prod_cstr.items():
            model_res = {}
            for res1, res2 in res_map.items():
                res_name = res1 + '_' + res2
                add_res = model.addResource(name=res_name, capacity={(0, "inf"): 1})
                model_res['virtual'] = {res1: add_res}
                model_res['virtual'].update({res2: add_res})
            model_res_grp[res_grp].update(model_res)

        return model_res_grp

    def add_res_capacity(self, res: Resource, capa_days) -> Resource:
        time_multiple = 1
        if self.time_unit == 'M':
            time_multiple = 60

        start_time = self.plant_start_hour
        end_time = self.plant_start_hour
        for i, time in enumerate(capa_days * self.schedule_weeks):
            start_time, end_time = util.calc_daily_avail_time(
                day=i, time=time*time_multiple, start_time=start_time, end_time=end_time
            )

            # Add the capacity
            res.addCapacity(start_time, end_time, 1)

            if i % 5 == 4:    # skip saturday & sunday
                start_time += self.sec_of_day * 3

        # Exception for over demand
        res.addCapacity(start_time + self.sec_of_day, 'inf', 1)

        return res

    def add_res_capacity_bak(self, res: Resource, avail_time) -> Resource:
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
                    duedate=due_date,    # duedate='inf',
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

                if self.cstr_cfg['apply_sim_prod_cstr']:
                    mode = self.add_virtual_res_on_mode(
                        mode=mode,
                        resource=resource,
                        model_res=model_res,
                        duration=duration
                    )

                # add mode list to activity
                act.addModes(mode)

        return act

    def add_virtual_res_on_mode(self, mode, resource, model_res, duration):
        virtual_dict = model_res.get('virtual', None)
        if virtual_dict:
            virtual_res = virtual_dict.get(resource, None)
            mode = self.add_resource(
                    mode=mode,
                    resource=virtual_res,
                    duration=duration
                )

        return mode

    # Add the specified resource which amount required when executing the mode
    @staticmethod
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
                duration_per_unit = self.default_res_duration

            if self.exec_cfg['verbose']:
                print(f"Item: {item_cd} does not have any resource duration.")
        else:
            duration_per_unit = item_res_duration.get(res_cd, None)
            if (duration_per_unit is None) or (duration_per_unit == 0):
                if self.except_cfg['miss_duration'] == 'add':
                    duration_per_unit = self.default_res_duration
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

    def set_job_change_activity(self, model: Model, activity: dict, model_res: dict):
        #
        act_by_res = self.get_res_to_dmd_list(model=model)

        # Set state
        state_map = self.set_state_map(data=act_by_res)
        model_state = self.set_model_state(model=model, data=act_by_res, state_map=state_map)

        for resource, act_list in act_by_res.items():
            res_mode = self.set_job_change_mode(
                resource=resource,
                act_list=act_list,
                state_map=state_map,
                model_res=model_res[self.res_to_res_grp[resource]],
                model_state=model_state[resource]
            )
            job_change_activity = self.add_job_change_activity(
                model=model,
                mode=res_mode,
                act_list=act_list,
                resource=resource
            )
            model = self.add_job_change_temporal(model, activity, job_change_activity)

        # self.conv_duplicate_job_change_activitiy(model=model)

        return model

    def choose_job_change_demand(self, dmd_list: list):
        # Resource group needed to consider job change time
        jc_res_grp_cand = list(self.job_change)

        dmd_grp_by_res_grp = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            if res_grp_cd in jc_res_grp_cand:
                # generate name
                act_name = util.generate_model_name(name_list=[dmd_id, item_cd, res_grp_cd])
                if res_grp_cd not in dmd_grp_by_res_grp:
                    dmd_grp_by_res_grp[res_grp_cd] = [act_name]
                else:
                    dmd_grp_by_res_grp[res_grp_cd].append(act_name)

        return dmd_grp_by_res_grp

    @staticmethod
    def add_job_change_temporal(model, activity, job_change_activity):
        for job_change_act in job_change_activity:
            model.addTemporal(job_change_activity[job_change_act], activity[job_change_act], 'CS')
            model.addTemporal(activity[job_change_act], job_change_activity[job_change_act], 'SC')

        return model

    @staticmethod
    def add_job_change_activity(model: Model, mode: dict, act_list: list, resource: str):
        job_change_activity = {}
        for act in act_list:
            job_change_activity[act] = model.addActivity(name=f'Setup[{act}@{resource}]', autoselect=True)
            for from_act, to_act in mode:
                if act == to_act:
                    job_change_activity[act].addModes(mode[(from_act, to_act)])

        return job_change_activity

    def set_job_change_mode(self, resource: str, act_list: list, state_map: dict, model_res: dict, model_state: dict):
        # Make act sequence list
        act_seq_list = self.make_act_sequence(resource=resource, act_list=act_list)

        res_mode = {}
        for from_act, to_act in act_seq_list:
            # Get job change time
            job_change_time = self.calc_job_change_time(
                res_grp=self.res_to_res_grp[resource],
                resource=resource,
                from_act=from_act,
                to_act=to_act
            )
            # Set job change mode
            res_mode[(from_act, to_act)] = Mode(
                name=f'Mode_setup[{from_act}|{to_act}|{resource}]',
                duration=job_change_time
            )

            res_mode[(from_act, to_act)].addState(
                state=model_state,
                fromValue=state_map[from_act],
                toValue=state_map[to_act]
            )
            if job_change_time != 0:
                res_mode[(from_act, to_act)].addResource(
                    resource=model_res[resource],
                    requirement={(0, job_change_time): 1}
                )

        return res_mode

    @staticmethod
    def make_act_sequence(resource: str, act_list: list) -> List[Tuple[str, str]]:
        act_seq_list = list(permutations(act_list, 2))
        act_seq_list += [(resource, act) for act in act_list]

        return act_seq_list

    def calc_job_change_time(self, res_grp: str, resource: str, from_act: str, to_act: str) -> int:
        if resource == from_act:
            jc_time = 0
        else:
            from_res = self.sku_to_type.get(from_act.split('@')[1], None)
            to_res = self.sku_to_type.get(to_act.split('@')[1], None)

            time_list = []
            for i, jc_type in enumerate(self.job_change_type):
                res_grp_jc = self.job_change.get(res_grp, None)
                if res_grp_jc:
                    time_dict = self.job_change[res_grp].get(jc_type, None)
                    if time_dict:
                        time = time_dict.get((from_res[i], to_res[i]), 0)
                    else:
                        time = 0
                else:
                    time = 0
                time_list.append(time)

            jc_time = max(time_list)

        return jc_time

    @staticmethod
    def set_state_map(data: dict) -> Dict[str, int]:
        state = {resource: i for i, resource in enumerate(sorted(data))}

        idx = len(data)
        for res_grp, demand_list in data.items():
            for demand in demand_list:
                state[demand] = idx
                idx += 1

        return state

    @staticmethod
    def set_model_state(model: Model, data: dict, state_map: dict):
        model_state = {}
        for resource in data:
            state = model.addState("state_" + resource)
            state.addValue(time=0, value=state_map[resource])
            model_state[resource] = state

        return model_state

    def check_fix_model_init_set(self, model: Model):
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

    def set_res_to_res_grp(self) -> None:
        res_grp = self.res_grp.copy()
        res_to_res_grp = {}
        for res_grp_cd, res_list in res_grp.items():
            for res_cd in res_list:
                res_to_res_grp[res_cd] = res_grp_cd

        self.res_to_res_grp = res_to_res_grp

    #####################
    # Save
    #####################
    def save_org_result(self) -> None:
        save_dir = os.path.join(self.save_path, 'opt', 'org', self.fp_version)
        util.make_dir(path=save_dir)

        result = open(os.path.join(save_dir, 'result_' + self.fp_name + '.txt'), 'w')
        with open(self.optseq_output_path, 'r') as f:
            for line in f:
                result.write(line)

        f.close()
        result.close()
