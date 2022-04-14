import common.util as util

from typing import  Dict
from optimize.optseq import Model, Mode, Parameters, Activity, Resource


class OptSeqModel(object):
    time_limit = 30
    make_span = False
    optput_flag = True

    res_type_capa_map = {
        'NOR': 1,
        'PPL': 2
    }

    def __init__(self, cstr_cfg: dict, plant: str, dmd_due: dict, item_res_grp_duration: dict):
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
        self.res_grp_default_duration = 1
        self.item_res_grp_duration_per_unit = item_res_grp_duration

    def setup_mode(self, data: list):
        job_change = {}
        for from_res_cd, to_res_cd, time in data:
            job_change[(from_res_cd, to_res_cd)] = Mode(
                name=f"Job_Change_{from_res_cd}_{to_res_cd}",
                duration=time
            )
            if time != 0:
                job_change[(from_res_cd, to_res_cd)].addResource()

    def init(self, dmd_list: list, res_grp_list: dict, job_change_list=[]):
        self.set_max_due_date()

        model = Model(name='lotte')

        if self.cstr_cfg['apply_job_change']:
            state = model.addState(name='job_change')
            state.addValue(time=0, value=0)

        # merge
        # res_grp_list = self.merge_res_grp(
        #     res_grp_list=res_grp_list,
        #     res_human_list=res_human_list
        # )

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
    def merge_res_grp(res_grp_list, res_human_list):
        for res_grp in res_human_list:
            if res_grp in res_grp_list:
                # Extend resource of people on original resource list
                res_grp_list[res_grp].extend(res_human_list[res_grp])

        return res_grp_list

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
                weight=1,    # Penalty per unit time when the work completion time is rate for delivery
                autoselect=self.cstr_cfg['apply_job_change']
            )

            # Calculate duration
            duration = self.calc_duration(item_cd, res_grp_cd, qty)

            # Set mode
            activity[act_name] = self.set_mode(
                act=activity[act_name],
                dmd_id=dmd_id,
                res_list=res_grp_list[res_grp_cd],
                model_res=model_res[res_grp_cd],
                duration=duration,
                res_grp_cd=res_grp_cd
            )

        return model

    def calc_duration(self, item_cd, res_grp_cd, qty) -> int:
        # Calculate duration
        item_res_grp_duration = self.item_res_grp_duration_per_unit.get(item_cd, None)

        if item_res_grp_duration is None:
            duration_per_unit = self.res_grp_default_duration
        else:
            duration_per_unit = item_res_grp_duration.get(res_grp_cd, self.res_grp_default_duration)

        duration = int(qty * duration_per_unit)

        return duration

    # Set work processing method
    def set_mode(self, act: Activity, dmd_id: str, res_list: list, model_res, duration: int,
                 res_grp_cd: str) -> Activity:
        for res_cd, capacity, unit_cd, res_type in res_list:
            if res_type == 'NOR':    # Check if resource is machine
                # Make each mode (set each available resource)
                mode = Mode(
                    # name=f'Mode[{res_cd}]',
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

                if self.add_res_people_yn:
                    res_to_peple_dict = self.res_human_map.get(res_grp_cd, None)
                    if res_to_peple_dict is not None:
                        people_list = res_to_peple_dict[res_cd]

                        # Add people
                        mode = self.add_resource_people(
                            mode=mode,
                            model_res=model_res,
                            people_list=people_list,
                            duration=duration
                        )

                # add mode list to activity
                act.addModes(mode)

        return act

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

    def add_resource_people(self, mode: Mode, model_res, people_list, duration: int):
        # Add resource
        for people in people_list:
            mode.addResource(    # ToDo: need to revise requirement
                resource=model_res[people],
                requirement={(0, duration): 1},    # requirement : gives the required amount of resources
                # requirement={(0, 1): 1}  # requirement : gives the required amount of resources
            )

        return mode

    def set_break(self):
        pass

    def set_parameter(self, model: Model) -> Model:
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span
        params.OutputFlag = self.optput_flag

        model.Params = params

        return model
