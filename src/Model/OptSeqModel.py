import common.util as util

from optimize.optseq import Model, Mode, Parameters


class OptSeqModel(object):
    time_limit = 100
    make_span = False
    optput_flag = True

    def __init__(self, dmd_due_date: dict, item_res_grp: dict, item_res_grp_duration: dict):
        self.dmd_due_date = dmd_due_date

        # Resource configuration
        self.item_res_grp = item_res_grp    # item -> available resource group list

        # Duration
        self.res_grp_default_duration = 0
        self.item_res_grp_duration = item_res_grp_duration   # Resource group -> Making time

        self.add_dummy_activity = True
        self.dummy_activity = None

    def init(self, dmd_list: list, res_grp_list):
        model = Model(name='lotte')

        if self.add_dummy_activity:
            self.set_dummy_activity(model=model)

        # Set resource
        model_res = self.set_resource(model=model, res_grp_list=res_grp_list)

        model = self.set_activity(
            model=model,
            dmd_list=dmd_list,
            model_res=model_res
        )

        model = self.set_parameter(model=model)

        return model

    @staticmethod
    def optimize(model: Model):
        model.optimize()
        print("")

    def set_dummy_activity(self, model: Model,):
        dummy_activity = model.addActivity(name='Act[dummy]', duedate="inf")
        dummy_mode = Mode(name='Mode[dummy]', duration=0)
        dummy_res = model.addResource(name='Res[dummy]', capacity=10000000)

        # Add dummy resource
        dummy_mode.addResource(resource=dummy_res)  # ToDo: need to revise
        dummy_mode.addResource(resource=dummy_res, requirement=1)    # ToDo: need to revise

        # Add dummy mode
        dummy_activity.addModes(dummy_mode)

        self.dummy_activity = dummy_activity

    def set_activity(self, model: Model, dmd_list: list, model_res):
        activity = {}
        for dmd_id, item_cd, qty, due_date in dmd_list:
            act_name = util.generate_name(dmd_id, item_cd)
            activity[act_name] = model.addActivity(name=f'Act[{act_name}]', duedate=due_date)

            # Set mode
            mode_list = self.set_mode(dmd_id=dmd_id, item_cd=item_cd, model_res=model_res)

            # add mode list to activity
            activity[act_name].addModes(mode_list)

            if self.add_dummy_activity:
                model.addTemporal(pred=activity[act_name], succ=self.dummy_activity, delay=0)

        return model

    def set_mode(self, dmd_id: str, item_cd: str, model_res) -> list:
        mode_list = []
        for res_grp in self.item_res_grp[item_cd]:
            # Make each mode (set each available resource)
            mode = Mode(
                name=f'Mode[{res_grp}]',
                duration=self.item_res_grp_duration[item_cd].get(res_grp, self.res_grp_default_duration)
            )

            # Add resource for each mode(resource)
            self.add_resource(mode=mode, dmd_id=dmd_id, item_cd=item_cd, res_grp=res_grp, model_res=model_res)

            mode_list.append(mode)

        return mode_list

    def set_resource(self, model: Model, res_grp_list):
        model_res_grp = {}
        for res_grp, res_list in res_grp_list.items():
            res_grp_capa = 0
            for res, capacity in res_list:
                res_grp_capa += capacity

            # Add the resource group
            add_res_grp = model.addResource(name=res_grp, capacity=res_grp_capa)
            model_res_grp[res_grp] = add_res_grp

        return model_res_grp

    # def set_resource(self, model: Model, res_grp_list):
    #     model_res_grp = {}
    #     for res_grp, res_list in res_grp_list.items():
    #         model_res_list = []
    #         for res, capacity in res_list:
    #             # Add the resource
    #             model_res = model.addResource(name=res, capacity=capacity)
    #             # Add the capacity of resource
    #             model_res.addCapacity(start=0, finish="inf", amount=capacity)   # ToDo: need to revise start, finish
    #             model_res_list.append(model_res)
    #
    #         model_res_grp[res_grp] = model_res_list
    #
    #     return model_res_grp

    def add_resource(self, mode: Mode, dmd_id: str, item_cd: str, res_grp: str, model_res: dict):
        # res_list = model_res[res_grp]
        # due_date = self.dmd_due_date[dmd_id][item_cd]
        #
        mode.addResource(resource=model_res[res_grp], requirement=1)  # ToDo: need to revise requirement
        # mode.addResource(resource=model_res[res_grp], requirement={(0, 9): 1})    # ToDo: need to revise requirement

        # resource list of resource group
        mode.addParallel(start=0, finish=100, maxparallel=list(model_res[res_grp].capacity.values())[0])

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