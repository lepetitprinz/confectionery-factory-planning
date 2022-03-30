from optimize.optseq import Model, Mode

import re
import datetime as dt


class Plan(object):
    def __init__(self, mst, mst_map, demand):
        self.mst = mst
        self.mst_map = mst_map
        self.demand = demand

        # Mapping
        self.oper_map = mst_map['oper']
        self.bom_map = mst_map['bom']
        self.demand_item = None

        # Model configuration
        self.time_limit = 10
        self.output_flag = True
        self.make_span = False

    def init(self, dmd_qty, bom_route, operation):
        self.demand_item = set(dmd_qty.index)

        # Initiate model
        model = Model()
        model.Params

        model, res = self._add_resource(model=model, res=self.mst_map['res'])
        # model, res = self._add_resource(model=model, res=self.mst_map['res'])
        model = self._add_activity(model=model, demand=dmd_qty, bom_route=bom_route, oper=operation, res=res)

        model.Params.TimeLimit = self.time_limit
        model.Params.OutputFlag = self.output_flag
        model.Params.Makespan = self.make_span

        return model

    def run(self, model: Model):
        # Optimization
        print('start optimization')
        model.optimize()
        print('end optimization')
        print("")

    def after_process(self, operation):
        f = open("optseq_output.txt", "r")

        obj_time = open("obj_time.csv", "w")
        obj_time.write("addbreObjVal,Time\n")
        while True:
            x = f.readline()
            if "objective" in x:
                obj = float(re.search(r"[0-9]+", x).group())
                time = float(re.search(r"[0-9]+\.[0-9]+\(s\)", x).group()[:-3])
                obj_time.write("{},{}\n".format(obj, time))
            if "sink,---," in x:
                break
        obj_time.close()
        f.readline()
        f.readline()
        line = []
        while True:
            x = f.readline()
            if "tardy activity" in x:
                break
            if x == "\n":
                continue
            line.append(x.split(","))
        f.close()

        self.bestsol_write_csv(line, oper=operation)

    @staticmethod
    def _add_resource(model: Model, res):
        resource = {r: model.addResource(name=r, capacity=int(res[r])) for r in res}

        # Todo: Exception
        resource['0000'] = model.addResource(name='0000', capacity=999999)

        return model, resource

    def _add_activity(self, model: Model, demand, bom_route, oper, res):
        act_map = {}
        mode = {}
        for item, count in zip(demand.index, demand.values):
            act_list, temporal = self._make_activity(
                bom_route=bom_route,
                item=item,
                count=count
            )
            for act in act_list:
                act_code = re.match(r"[A-Z]+", act).group()
                if act_code in self.demand_item:
                    index = int(re.search(r"(?<={})[0-9]+".format(act_code), act).group())
                    dmd = self.demand[self.demand['code'] == act_code].reset_index()
                    act_map[act] = model.addActivity(name=act, duedate=dmd.loc[index, 'minutes'])
                else:
                    act_map[act] = model.addActivity(name=act)

                # Exception
                if self.oper_map.get(act_code, None) is None:
                    schd_time = 0
                else:
                    schd_time = int(self.oper_map[act_code]['schd_time'].sum())
                mode[act] = Mode(name="mode_" + act, duration=schd_time)
                schedule_time = oper[oper['item_cd'] == self.bom_map['item_code_rev_map'][act_code]]['schd_time']

                for index in range(len(schedule_time)):
                    mode[act].addBreak(int(schedule_time[:index].sum()), int(schedule_time[:index].sum()), "inf")

                if self.oper_map.get(act_code, None) is not None:
                    for res_code, start, end in self._make_resource(self.oper_map[act_code]):
                        mode[act].addResource(res[res_code], {(start, end): 1})
                else:
                    # Todo: Exception
                    mode[act].addResource(res['0000'], {(0, 0): 1})

                act_map[act].addModes(mode[act])

            for f, l in temporal:
                model.addTemporal(act_map[f], act_map[l])

        return model

    def _make_activity(self, bom_route, item, count=1, act_list=None, act_name=None, temporal=None):
        if act_list is None:
            act_list = []
            temporal = []

        for i in range(count):
            if act_name is None:
                act_name = item + str(i)
                act_list.append(act_name)
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

    def _make_resource(self, df):
        res = []
        df = df.sort_values("operation_no")
        start = 0
        for i in df.index:
            res.append((df['wc_cd'][i], int(start), int(start + df['schd_time'][i])))
            start += df['schd_time'][i]

        return res

    def bestsol_write_csv(self, line, oper):
        f = open("bestsolution.csv", "w")
        f.write("Act,Code,Num,Product,Resource,Start,End,StartTime,EndTime\n")
        now = dt.datetime.now()
        for i in line:
            actname = i[0]
            actno = re.search(r"[0-9]+$", i[0]).group()
            item = re.match(r"[A-Z]+", actname).group()
            times = i[2]
            acts = [tuple(i.split("--")) for i in re.findall(r"[0-9]+\-\-[0-9]+", times)]
            sub_df = oper[oper.ITEM_CD == self.bom_map['item_code_rev_map'][item]]
            op_no = list(sub_df.OPERATION_NO)
            for s, e in acts:
                s, e = int(s), int(e)
                time = e - s
                while time:
                    no = op_no.pop(0)
                    now_df = sub_df[sub_df.OPERATION_NO == no]
                    sc_time = int(now_df.SCHD_TIME)
                    wc_cd = now_df.WC_CD.reset_index(drop=True)[0]
                    e = s + sc_time
                    stime = now + dt.timedelta(minutes=s)
                    etime = now + dt.timedelta(minutes=e)
                    f.write(
                        "{},{},{},{},{},{},{},{},{}\n".format(actname + "_" + str(no),
                                                              item, actno, self.bom_map['item_code_rev_map'][item],
                                                              wc_cd,
                                                              s, e, stime, etime))
                    s += sc_time
                    time -= sc_time

        f.close()
