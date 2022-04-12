import common.util as util

from optimize.optseq import Model, Mode, Parameters


class OptSeqModel(object):
    time_limit = 10
    make_span = False
    optput_flag = True

    res_type_capa_map = {
        'NOR': 1,
        'PPL': 2
    }

    def __init__(self, plant: str, dmd_due: dict, res_info: dict):
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
        self.sec_of_day = 86400
        self.capa_type = 'daily_capa'
        self.res_start_time_of_day = 0

        # Duration instance attribute
        self.res_grp_default_duration = 1
        self.item_res_grp_duration_per_unit = res_info['plant_item_res_grp_duration'][plant]

    def init(self, dmd_list: list, res_grp_list: dict):
        self.set_max_due_date()

        model = Model(name='lotte')

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
        print("")

    # Set work defined
    def set_activity(self, model: Model, dmd_list: list, res_grp_list: dict, model_res):
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            act_name = util.generate_name(name_list=[dmd_id, item_cd, res_grp_cd])
            activity[act_name] = model.addActivity(
                name=f'Act[{act_name}]',    # Work name
                duedate=due_date,
                weight=1    # Penalty per unit time when the work completion time is rate for delivery
            )

            # Calculate duration
            duration = self.calc_duration(item_cd, res_grp_cd, qty)

            # Set mode
            activity[act_name] = self.set_mode(
                act=activity[act_name],
                res_list=res_grp_list[res_grp_cd],
                model_res=model_res[res_grp_cd],
                duration=duration,
                res_grp_cd=res_grp_cd
            )

        return model

    def calc_duration(self, item_cd, res_grp_cd, qty):
        # Calculate duration
        item_res_grp_duration = self.item_res_grp_duration_per_unit.get(item_cd, None)

        if item_res_grp_duration is None:
            duration_per_unit = self.res_grp_default_duration
        else:
            duration_per_unit = item_res_grp_duration.get(res_grp_cd, self.res_grp_default_duration)

        duration = qty * duration_per_unit

        return duration

    # Set work processing method
    def set_mode(self, act,  res_list: list, model_res, duration: int, res_grp_cd: str) -> list:
        for res_cd, capacity, res_type in res_list:
            if res_type == 'NOR':    # Check if resource is machine
                # Make each mode (set each available resource)
                mode = Mode(
                    name=f'Mode[{res_cd}]',
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

    def set_resource(self, model: Model, res_grp_list):
        model_res_grp = {}
        for res_grp, res_list in res_grp_list.items():
            model_res = {}
            for res, capacity, unit_cd, res_type in res_list:
                # Add the resource
                add_res = model.addResource(name=res)
                add_res = self.set_capacity(res=add_res, capacity=capacity, unit_cd=unit_cd)
                model_res[res] = add_res
            model_res_grp[res_grp] = model_res

        return model_res_grp

    def set_capacity(self, res, capacity, unit_cd):
        if self.capa_type == 'daily_capa':
            capa = 0
            if unit_cd == 'MIN':
                capa = capacity * 60
            elif unit_cd == 'SEC':
                capa = capacity

            for i in range(self.max_due_day + 1):
                start_time = i * self.sec_of_day
                res.addCapacity(start_time, start_time + capa, 1)

        return res

    # Add the specified resource which amount required when executing the mode
    def add_resource(self, mode: Mode, resource, duration: int):
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

    def set_parameter(self, model: Model):
        # Set parameters
        params = Parameters()
        params.TimeLimit = self.time_limit
        params.Makespan = self.make_span
        params.OutputFlag = self.optput_flag

        model.Params = params

        return model
