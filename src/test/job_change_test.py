from optimize.optseq import Model, Mode, Parameters, Activity, Resource

m1 = Model()
duration = {"D": 30, "E": 30, "F": 30}
setup = {("D", "E"): 10, ("E", "D"): 50, ("E", "F"): 10, ("F", "E"): 10,
         ("start", "D"): 0, ("start", "E"): 10, ("start", "F"): 0,
         ("D", "F"): 0, ("F", "D"): 0}
s = {"D": 1, "E": 2, "F": 3, "start": 0}

rs = m1.addResource("line1", 1)

act = {}
mode = {}
for i in duration:
    act[i] = m1.addActivity(f"Act_{i}")
    mode[i] = Mode(f"Mode_{i}", duration[i])
    mode[i].addResource(rs, {(0, "inf"): 1})
    act[i].addModes(mode[i])

s1 = m1.addState("Setup_State")
s1.addValue(time=0, value=s["start"])
# setup mode
mode_setup = {}
for (i, j) in setup:
    mode_setup[i, j] = Mode(f"Mode_setup_{i}_{j}", setup[i, j])
    mode_setup[i, j].addState(s1, s[i], s[j])
    if setup[i, j] != 0:
        mode_setup[i, j].addResource(rs, {(0, setup[i, j]): 1})

    # print (i,j,s[i],s[j],mode_setup[i,j])

act_setup = {}
for k in duration:
    act_setup[k] = m1.addActivity(f"Setup_{k}", autoselect=True)
    for (i, j) in setup:
        if k == j:
            act_setup[k].addModes(mode_setup[i, j])

# temporal (precedense) constraints
for j in act_setup:
    m1.addTemporal(act_setup[j], act[j], "CS")
    m1.addTemporal(act[j], act_setup[j], "SC")

print("")