from optimize.optseq import Model, Mode, Parameters, Activity, Resource

model = Model()

duration_dict = {
    'A': {"A1": 100, "A2": 100},
    'B': {"B1": 100, "B2": 100},
    'C': {"C1": 100, "C2": 100}
    }

setup = {
    'A': {
        ("A1", "A2"): 10, ("A2", "A1"): 50,
        ("start1", "A1"): 0, ("start1", "A2"): 0,
        # ("A1", "start1"): 0, ("A2", "start1"): 0,
    },
    'B': {
        ("B1", "B2"): 30, ("B2", "B1"): 40,
        ("start2", "B1"): 0, ("start2", "B2"): 0,
        # ("B1", "start2"): 0, ("B2", "start2"): 0,
    },
}
setup_list = ['A', 'B']

state_dict = {"start1": 0, "start2": 1,  "A1": 3, "A2": 4, "B1": 5, "B2": 6}

res_dict = {
    'A': model.addResource('line1', 1),
    'B': model.addResource('line2', 1),
    'C': model.addResource('line3', 1),
}

act = {}
mode = {}
state = {}
for i, (line, dur) in enumerate(duration_dict.items()):
    for demand, d in dur.items():
        act[demand] = model.addActivity(f'Act[{demand}]')
        mode[demand] = Mode(f'Mode[{demand}]', int(d))
        mode[demand].addResource(res_dict[line], {(0, "inf"): 1})
        act[demand].addModes(mode[demand])

    if line in setup_list:
        s = model.addState("state" + line)
        s.addValue(time=0, value=i)
        state[line] = s

mode_setup = {}
for line, time_map in setup.items():
    if line in setup_list:
        for (from_time, to_time), duration in time_map.items():
            mode_setup[(from_time, to_time)] = Mode(f'Mode_setup[{from_time}_{to_time}]', duration)
            mode_setup[(from_time, to_time)].addState(state[line], state_dict[from_time], state_dict[to_time])
            if duration != 0:
                mode_setup[(from_time, to_time)].addResource(res_dict[line], {(0, duration): 1})

act_setup = {}
for line, dur in duration_dict.items():
    if line in setup_list:
        for demand, duration in dur.items():
            act_setup[demand] = model.addActivity(f"Setup[{demand}]", autoselect=True)
            setup_line = setup.get(line, None)
            if setup_line:
                for from_time, to_time in setup_line:
                    if demand == to_time:
                        act_setup[demand].addModes(mode_setup[(from_time, to_time)])

for demand in act_setup:
    model.addTemporal(act_setup[demand], act[demand], 'CS')
    model.addTemporal(act[demand], act_setup[demand], 'SC')

model.Params.TimeLimit = 10
model.Params.Makespan = True
model.Params.OutputFlag = True
# model.Params.Backtruck = 10

model.optimize()