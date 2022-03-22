from optimize.optseq import Model, Mode
from optimize.optmodule import *

import pandas as pd
import numpy as np

bom = pd.read_csv("BOM.csv")
wcdf = pd.read_csv("WC.csv")
oper = pd.read_csv("Operation.csv")
demand = pd.read_csv("Demand.csv")
itemdf = pd.read_csv("ITEM.csv")

demand["deadline"] = pd.to_datetime(demand["DUEDATE"].astype(str))
start = demand["deadline"].min()
demand["days"] = (demand["deadline"] - start)
demand["minutes"] = demand["days"] / np.timedelta64(1, "m")
itemlist = list(set(bom.PARENT_ITEM) | set(bom.CHILD_ITEM))
item_code, code_item = {}, {}
for i in range(len(itemlist)):
    n = base(i, 10)
    item_code[itemlist[i]] = n
    code_item[n] = itemlist[i]
demand["ITEM_CODE"] = [item_code[demand.ITEM_CD[i]] for i in demand.index]
oper["ITEM_CODE"] = [item_code[oper.ITEM_CD[i]] for i in oper.index]
oper["SCHD_TIME"] = list(map(lambda x: np.ceil(x), oper.SCHD_TIME))
opdict = dict(list(map(lambda x: (item_code[x[0]], x[1]), oper.groupby("ITEM_CD"))))
wc = dict([(wcdf.WC_CD[i], wcdf.WC_NUM[i]) for i in wcdf.index])
demandlist = [(i[0], i[1].QTY.sum()) for i in demand.groupby("ITEM_CODE")]

# debugging #001
print("debugging #001")
print(demandlist)

demandset = set(demand.ITEM_CODE)
BOM1, BOM = make_bom(bom, item_code)

model = Model()
resource, act, mode = {}, {}, {}
errorlist = []
for r in wc:
    resource[r] = model.addResource(name=r, capacity=int(wc[r]))
for item, repeat in demandlist:
    actlist, temporal = make_activities(BOM, item, repeat)
    for a in actlist:
        actcode = re.match(r"[A-Z]+", a).group()
        if actcode in demandset:
            index = int(re.search(r"(?<={})[0-9]+".format(actcode), a).group())
            ddf = demand[demand.ITEM_CODE == actcode].reset_index()

            # debugging #002
            print("debugging #002")
            print(ddf)

            ddf.minutes[index]
            act[a] = model.addActivity(name=a, duedate=int(ddf.minutes[index]))
        else:
            act[a] = model.addActivity(name=a)
        mode[a] = Mode(name="mode_" + a, duration=int(opdict[actcode].SCHD_TIME.sum()))
        SCHD_TIME = oper[oper.ITEM_CD == code_item[actcode]].SCHD_TIME
        for index in range(len(SCHD_TIME)):
            time = int(SCHD_TIME[:index].sum())
            mode[a].addBreak(int(SCHD_TIME[:index].sum()), int(SCHD_TIME[:index].sum()), "inf")
        for r, s, e in make_reslist(opdict[actcode]):
            mode[a].addResource(resource[r], {(s, e): 1})

        act[a].addModes(mode[a])

    for f, l in temporal:
        model.addTemporal(act[f], act[l])
model.Params.TimeLimit = 1
model.Params.OutputFlag = True
model.Params.Makespan = False

model.optimize()

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
L = []
while True:
    x = f.readline()
    if "tardy activity" in x:
        break
    if x == "\n":
        continue
    L.append(x.split(","))
f.close()

bestsol_write_csv(L, oper, code_item)
