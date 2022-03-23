from optimize.optseq import Model, Mode

import re


class Plan(object):
    def __init__(self, mst, mst_map):
        self.mst = mst
        self.mst_map = mst_map

        # Model configuration
        self.time_limit = 1
        self.output_flag = True
        self.make_span = False

    def init(self, demand, bom_route):
        model = Model()

        model, res = self._add_resource(model=model, res=self.mst_map['res'])
        model = self._add_activity(model=model, demand=demand, bom_route=bom_route)

        model.Params.TimeLimit = self.time_limit
        model.Params.OutputFlag = self.output_flag
        model.Params.Makespan = self.make_span

        return model

    def run(self, demand, res, bom_route):
        model = self.init(demand=demand, res=res, bom_route=bom_route)

        # Optimization
        model.optimize()

    @staticmethod
    def _add_resource(model: Model, res):
        resource = {r: model.addResource(name=r, capacity=int(res[r])) for r in res}

        return model, resource

    def _add_activity(self, model: Model, demand, bom_route):
        for item, count in zip(demand.index, demand.values):
            act_list, temporal = self._make_activity(
                bom_route=bom_route,
                item=item,
                count=count
            )
            for act in act_list:
                act_code = re.match(r"[A-Z]+", act).group()

    def _make_activity(self, bom_route, item, count=1, act_list=None, act_name=None, temporal=None):
        if act_list is None:
            act_list = []
            temporal = []

        for i in range(count):
            if act_name is None:
                act_name = item + str(i)
            else:
                if item + str(i-1) in act_name:
                    act_name = act_name[act_name.index(item + str(i - 1)) + len(item + str(i - 1)) + 1:]
                if act_name is None:
                    act_name = item + str(i)
                else:
                    temporal.append((item + str(i) + '_' + act_name, act_name))
                    act_name = item + str(i) + "_" + act_name

                act_list.append(act_name)

            for c in bom_route.predecessors(item):
                self._make_activity(
                    bom_route=bom_route,
                    item=c,
                    count=bom_route[c][item]['rate'],
                    act_list=act_list,
                    act_name=act_name,
                    temporal=temporal
                )

        return act_list, temporal