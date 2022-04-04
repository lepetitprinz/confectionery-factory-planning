import common.util as util

from optimize.optseq import Model, Mode, Parameters


class OptSeqModel(object):
    time_limit = 10
    make_span = True
    optput_flag = True

    def __init__(self, dmd_due_date: dict, item_res_grp: dict, item_res_grp_duration: dict,
                 res_to_people_by_plant: dict):
        self.dmd_due_date = dmd_due_date
        self.max_due_date = ''

        # Resource instance attribute
        self.item_res_grp = item_res_grp    # item -> available resource group list
        self.res_to_people_by_plant = res_to_people_by_plant
        self.add_res_people_yn = True

        # Duration instance attribute
        self.res_grp_default_duration = 1
        self.item_res_grp_duration_per_unit = item_res_grp_duration   # Resource group -> Making time

        # Dummy configuration instance attribute
        self.add_dummy_activity = False    # True / False
        self.dummy_activity = None
        self.dummy_capacity = 100000

    def set_max_due_date(self):
        due_list = []
        for sku_due in self.dmd_due_date.values():
            for due_date in sku_due.values():
                due_list.append(due_date)

        due_list = sorted(due_list, reverse=True)

        self.max_due_date = due_list[0]

    def merge_res_grp(self, res_grp_list, res_grp_people_list):
        for res_grp in res_grp_people_list:
            if res_grp in res_grp_list:
                # Extend resource of people on original resource list
                res_grp_list[res_grp].extend(res_grp_people_list[res_grp])

        return res_grp_list

    def init(self, dmd_list: list, res_grp_list: dict, res_grp_people_list: dict):
        self.set_max_due_date()

        model = Model(name='lotte')

        if self.add_dummy_activity:
            self.set_dummy_activity(model=model)

        # merge
        res_grp_all = self.merge_res_grp(res_grp_list=res_grp_list, res_grp_people_list=res_grp_people_list)

        # Set resource
        model_res = self.set_resource(model=model, res_grp_list=res_grp_all)

        model = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            res_grp_list=res_grp_all,
            model_res=model_res
        )

        model = self.set_parameter(model=model)

        return model

    @staticmethod
    def optimize(model: Model):
        model.optimize()
        print("")

    def set_dummy_activity(self, model: Model,):
        dummy_activity = model.addActivity(name='Act[dummy]', duedate=23614158)

        # Define the mode
        dummy_mode = Mode(name='Mode[dummy]', duration=0)

        # Add dummy resource
        dummy_mode.addResource(    # ToDo: need to revise
            resource=model.addResource(name='Res[dummy]', capacity=self.dummy_capacity),
            requirement={(0, 1): 1}
        )
        # dummy_mode.addParallel(    # ToDo: need to revise
        #     start=0,
        #     finish=1000000,
        #     maxparallel=self.dummy_capacity
        # )

        # Add dummy mode
        dummy_activity.addModes(dummy_mode)

        self.dummy_activity = dummy_activity

    # Set work defined
    def set_activity(self, model: Model, dmd_list: list, res_grp_list: list, model_res):
        activity = {}
        for dmd_id, item_cd, res_grp_cd, qty, due_date in dmd_list:
            act_name = util.generate_name(name_list=[dmd_id, item_cd, res_grp_cd])
            activity[act_name] = model.addActivity(
                name=f'Act[{act_name}]',    # Work name
                # duedate=due_date,           # Delivery date of work
                duedate=self.max_due_date,
                weight=1                    # Penalty per unit time when the work completion time is rate for delivery
            )

            # Calculate duration
            duration_per_unit = self.item_res_grp_duration_per_unit[item_cd].get(
                res_grp_cd, self.res_grp_default_duration
            )
            duration = qty * duration_per_unit

            # Set mode
            mode_list = self.set_mode(
                res_list=res_grp_list[res_grp_cd],
                model_res=model_res[res_grp_cd],
                duration=duration,
                res_grp_cd=res_grp_cd
            )

            # add mode list to activity
            activity[act_name].addModes(mode_list)

            model.addTemporal(pred="source", succ=activity[act_name], delay=0)
            model.addTemporal(pred=activity[act_name], succ="sink", delay=0)

            # if self.add_dummy_activity:
            #     model.addTemporal(pred=activity[act_name], succ=self.dummy_activity, delay=0)

        return model

    # Set work processing method
    def set_mode(self, res_list: list, model_res, duration: int, res_grp_cd: str) -> list:
        mode_list = []
        for res_cd, capacity, res_type in res_list:
            if res_type == 'NOR':
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
                    people_list = self.res_to_people_by_plant[res_grp_cd][res_cd]

                    mode = self.add_resource_people(
                        mode=mode,
                        model_res=model_res,
                        people_list=people_list,
                        duration=duration
                    )

                mode_list.append(mode)

        return mode_list

    def set_resource(self, model: Model, res_grp_list):
        model_res_grp = {}
        for res_grp, res_list in res_grp_list.items():
            model_res = {}
            for res, capacity, res_type in res_list:
                # Add the resource
                add_res = model.addResource(res, {(0, self.max_due_date): 1})

                # Add the capacity of resource    # ToDo: need to revise start, finish
                # add_res.addCapacity(start=0, finish=self.max_due_date, amount=1)

                model_res[res] = add_res
            model_res_grp[res_grp] = model_res

        return model_res_grp

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