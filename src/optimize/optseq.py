# Python Interface for the Resource Constrained Scheduling Solver OptSeq
# file name  optseq3.py
# ver. 2.5 Copyright Log Opt Co., Ltd.

import sys
import copy
import platform

if platform.system() == "Windows":
    optseq = "OptSeq"
else:
    optseq = "./OptSeq"

if int(sys.version_info[0]) <= 2:
    import string
#    _trans = string.maketrans("-+*/'(){} ^=<>$&#?,","_"*19)
else:
    _trans = str.maketrans("-+*/'(){} ^=<>$&#?,", "_" * 19)


class Parameters():
    """
    OptSeq parameter class to control the operation of OptSeq.
    
    @param  TimeLimit: 	Limits the total time expended (in seconds). Positive integer. Default=600.
    @param  OutputFlag: Controls the output log. Boolean. Default=False (0).
    @param  RandomSeed: Sets the random seed number. Integer. Default=1.
    @param  ReportInterval: Controls the frequency at which log lines are printed (iteration number). Default=1073741823.
    @param  Backtruck: Controls the maximum backtrucks. Default=1000.
    @param  MaxIteration: Sets the maximum numbers of iterations. Default=1073741823.
    @param  Initial: =True if the user wants to set an initial activity list. Default = False.
            Note that the file name of the activity list must be "optseq_best_act_data.txt."
    @param  Tenure: Controls a parameter of tabu search (initial tabu tenure). Default=0.
    @param  Makespan: Sets the objective function.
            Makespan is True if the objective is to minimize the makespan (maximum completion time),
            is False otherwise, i.e., to minimize the total weighted tardiness of activities.
            Default=False.
    """

    def __init__(self):
        self.TimeLimit = 600
        self.OutputFlag = 0  # off
        self.RandomSeed = 1
        self.ReportInterval = 1073741823
        self.Backtruck = 1000
        self.MaxIteration = 1073741823
        self.Initial = False
        self.Tenure = 1
        self.Makespan = False


class Mode():
    def __init__(self, name="", duration=0):
        """
        OptSeq mode class.

            - Arguments:
                - name: Name of mode (sub-activity). 
                Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                Also you cannot use "dummy" for the name of a mode.
                - duration(optional): Processing time of mode. Default=0.

            - Attbibutes: 
                - requirement: Dictionary that maps a pair of resource name and resource type (rtype) to requirement dictionary.
                Requirement dictionary maps intervals (pairs of start time and finish time) to amounts of requirement.
                Resource type (rtype) is None (standard resource type), "break" or "max."  
                - breakable: Dictionary that maps breakable intervals to maximum brek times.
                - paralel:  Dictionary that maps parallelable intervals to maximum parallel numbers. 
                - state: Dictionary that maps states to the tuples of values.
        """
        if name == "dummy":
            print("source and sink cannnot be used as an activity name")
            raise NameError
        self.name = name
        self.duration = duration
        self.requirement = {}
        self.breakable = {}
        self.parallel = {}
        self.state = {}

    def __str__(self):
        ret = [" duration {0} ".format(self.duration)]

        if self.requirement != {}:
            for (r, rtype) in self.requirement:
                for (interval, cap) in self.requirement[(r, rtype)].items():
                    (s, t) = interval
                    if rtype == "max":
                        ret.append(" {0} max interval {1} {2} requirement {3} ".format(r, s, t, cap))
                    elif rtype == "break":
                        ret.append(" {0} interval break {1} {2} requirement {3} ".format(r, s, t, cap))
                    elif rtype == None:
                        ret.append(" {0} interval {1} {2} requirement {3} ".format(r, s, t, cap))
                    else:
                        print("resource type error")
                        raise TypeError

        # break
        if self.breakable != {}:
            for (interval, cap) in self.breakable.items():
                (s, t) = interval
                if cap == "inf":
                    ret.append(" break interval {0} {1} ".format(s, t))
                else:
                    ret.append(" break interval {0} {1} max {2} ".format(s, t, cap))

        # parallel
        if self.parallel != {}:
            for (interval, cap) in self.parallel.items():
                (s, t) = interval
                if cap == "inf":
                    ret.append(" parallel interval {0} {1} ".format(s, t))
                else:
                    ret.append(" parallel interval {0} {1} max {2} ".format(s, t, cap))

        # state
        if self.state != {}:
            for s in self.state:
                (f, t) = self.state[s]
                ret.append(" {0} from {1} to {2} ".format(s, f, t))

        return " \n".join(ret)

    def addState(self, state, fromValue=0, toValue=0):
        """
        Adds a state change information to the mode.

            - Arguments:
                - state: State object to be added to the mode.
                - fromValue: the value from which the state changes by the mode
                - toValue:  the value to which the state changes by the mode
                
            - Example usage:

            >>> mode.addState(state1,0,1)
            
            defines that state1 is changed from 0 to 1. 

        """
        if type(fromValue) != type(1) or type(toValue) != type(1):
            print("time and value of the state {0} must be integer".format(self.name))
            raise TypeError
        else:
            self.state[state.name] = (fromValue, toValue)

    def addResource(self, resource, requirement={}, rtype=None):
        """
        Adds a resource to the mode.

            - Arguments:
                - resurce: Resource object to be added to the mode.
                - requirement: Dictionary that maps intervals (pairs of start time and finish time) to amounts of requirement.
                               It may be an integer; in this case, requirement is converted into the dictionary {(0,"inf"):requirement}.
                - rtype (optional): Type of resource to be added to the mode.
                None (standard resource type; default), "break" or "max."
            
            - Example usage:

            >>> mode.addResource(worker,{(0,10):1})
            
            defines worker resource that uses 1 unit for 10 periods. 

            >>> mode.addResource(machine,{(0,"inf"):1},"break")

            defines machine resource that uses 1 unit during break periods. 

            >>> mode.addResource(machine,{(0,"inf"):1},"max")

            defines machine resource that uses 1 unit during parallel execution. 
        """
        if type(requirement) == type(1):
            requirement = {(0, "inf"): requirement}

        if type(resource.name) != type("") or type(requirement) != type({}):
            print("type error in adding a resource {0} to activity".format(resource.name, self.name))
            raise TypeError
        elif rtype == None or rtype == "break" or rtype == "max":
            if (resource.name, rtype) not in self.requirement:
                self.requirement[(resource.name, rtype)] = {}  # generate an empty dic.
            data = copy.deepcopy(self.requirement[(resource.name, rtype)])
            data.update(requirement)
            self.requirement[(resource.name, rtype)] = data
        else:
            print("rtype must be None or break or max")
            raise NameError

    def addBreak(self, start=0, finish=0, maxtime="inf"):
        """
        Sets breakable information to the mode.

            - Arguments:
                - start(optional): Earliest break time. Non-negative integer. Default=0.
                - finish(optional): Latest break time.  Non-negative integer or "inf." Default=0.
                    Interval (start,finish) defines a possible break interval.
                - maxtime(optional): Maximum break time. Non-negative integer or "inf." Default="inf."
           
            - Example usage:

            >>> mode.addBreak(0,10,1}

            defines a break between (0,10) for one period.      
        """
        data = copy.deepcopy(self.breakable)
        data.update({(start, finish): maxtime})
        self.breakable = data

    def addParallel(self, start=1, finish=1, maxparallel="inf"):
        """
        Sets parallel information to the mode.

            - Arguments:
                - start(optional): Smallest job index executable in parallel. Positive integer. Default=1.
                - finish(optional): Largest job index executable in parallel. Positive integer or "inf." Default=1.
                - maxparallel(optional): Maximum job numbers executable in parallel. Non-negative integer or "inf." Default="inf."
           
            - Example usage:

            >>> mode.addParallel(1,1,2}               
        """
        data = copy.deepcopy(self.parallel)
        data.update({(start, finish): maxparallel})
        self.parallel = data


class Activity():
    ID = 0

    def __init__(self, name="", duedate="inf", weight=1, autoselect=False):
        """
        OptSeq activity class.

        You can create an activity object by adding an activity to a model (using model.addActivity)
        instead of by using an Activity constructor.    

            - Arguments:
                - name: Name of activity. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                    Also you cannot use "source" and "sink" for the name of an activity.
                - duedate(optional): Duedate of activity. A non-nagative integer or string "inf."
                - weight(optional): Panalty of one unit of tardiness. Positive integer.
                - autoselect(optional): True or False flag that indicates the activity selects the mode automatically or not.
        """

        if name == "source" or name == "sink":
            print("source and sink cannnot be used as an activity name")
            raise NameError
        if name == "" or name == None:
            name = "__a{0}".format(Activity.ID)
            Activity.ID += 1
        # convert illegal characters into _ (underscore)
        self.name = str(name).translate(_trans)
        self.duedate = duedate
        self.weight = weight
        self.autoselect = autoselect
        self.modes = []  # list of mode objects

    def __str__(self):
        ret = ["activity {0}".format(self.name)]
        if self.duedate != "inf":
            ret.append(" duedate {0} ".format(self.duedate))
            ret.append(" weight {0} ".format(self.weight))
        if len(self.modes) == 1:  # single mode
            ret.append(" mode {0} ".format(self.modes[0]))  # print mode information
            # ret.append(str(self.modes[0]))
        elif len(self.modes) >= 2:
            if self.autoselect == True:
                ret.append(" autoselect ")
            for m in self.modes:  # multiple modes
                ret.append(" {0} ".format(m.name))  # print mode names
            # ret+="\n"
        else:
            ret.append(" no mode ")
        return " \n".join(ret)

    def addModes(self, *modes):
        """
        Adds a mode or modes to the activity.

            - Arguments:
                - modes: One or more mode objects.
       
            - Example usage:

            >>> activity.addModes(mode1,mode2)               
        """
        for mode in modes:
            self.modes.append(mode)


class Resource():
    ID = 0

    def __init__(self, name="", capacity={}, rhs=0, direction="<=", weight="inf"):
        """
        OptSeq resource class.

            - Arguments:
                - name: Name of resource. 
                Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                - capacity (optional): Capacity dictionary of the renewable (standard) resource.
                    Capacity dictionary maps intervals (pairs of start time and finish time) to amounts of capacity.
                    If it is given by a positive integer, it is converted into the dictionay {(0,"inf"):capacity}.
                - rhs (optional): Right-hand-side constant of nonrenewable resource constraint. 
                - direction (optional): Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or ">=".
                - weight (optional): Weight of nonrenewable resource to compute the penalty for violating the constraint. Non-negative integer or "inf" (default).

            - Attbibutes:
                - capacity: Capacity dictionary of the renewable (standard) resource.
                - rhs: Right-hand-side constant of nonrenewable resource constraint. 
                - direction: Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or ">=". 
                - terms: List of terms in left-hand-side of nonrenewable resource. 
                   Each term is a tuple of coeffcient,activity and mode.
                - weight: Weight of nonrenewable resource to compute the penalty for violating the constraint. Non-negative integer or "inf" (default).
     
        """
        if name == "" or name == None:
            name = "__r{0}".format(Resource.ID)
            Resource.ID += 1
        # convert illegal characters into _ (underscore)
        self.name = str(name).translate(_trans)

        if type(capacity) == type(1):
            capacity = {(0, "inf"): capacity}

        self.capacity = capacity
        self.rhs = rhs
        self.direction = direction
        self.terms = []
        self.weight = weight

    def __str__(self):
        ret = []
        if self.capacity != {}:
            ret.append("resource {0} ".format(self.name))
            capList = []
            for (interval, cap) in self.capacity.items():
                (s, t) = interval
                capList.append((s, t, cap))
            capList.sort()
            for (s, t, cap) in capList:
                ret.append(" interval {0} {1} capacity {2} ".format(s, t, cap))
            # ret.append("\n")
        return " \n".join(ret)

    def addCapacity(self, start=0, finish=0, amount=1):
        """
        Adds a capacity to the resource.

            - Arguments:
                - start(optional): Start time. Non-negative integer. Default=0.
                - finish(optional): Finish time. Non-negative integer. Default=0.
                 Interval (start,finish) defines the interval during which the capacity is added.
                - amount(optional): The amount to be added to the capacity. Positive integer. Default=1.
        
            - Example usage:

            >>> manpower.addCapacity(0,5,2)
        """

        data = copy.deepcopy(self.capacity)
        data.update({(start, finish): amount})
        self.capacity = data

    def printConstraint(self):
        """
            Returns the information of the linear constraint.
            
            The constraint is expanded and is shown in a readable format.
        """

        f = ["nonrenewable weight {0} ".format(self.weight)]
        if self.direction == ">=" or self.direction == ">":
            for (coeff, var, value) in self.terms:
                f.append("{0}({1},{2}) ".format(-coeff, var.name, value.name))
            f.append("<={0} \n".format(-self.rhs))
        elif self.direction == "==" or self.direction == "=":
            for (coeff, var, value) in self.terms:
                f.append("{0}({1},{2}) ".format(coeff, var.name, value.name))
            f.append("<={0} \n".format(self.rhs))
            f.append("nonrenewable weight {0} ".format(self.weight))
            for (coeff, var, value) in self.terms:
                f.append("{0}({1},{2}) ".format(-coeff, var.name, value.name))
            f.append("<={0} \n".format(-self.rhs))
        else:
            for (coeff, var, value) in self.terms:
                f.append("{0}({1},{2}) ".format(coeff, var.name, value.name))
            f.append("<={0} \n".format(self.rhs))

        return "".join(f)

    def addTerms(self, coeffs=[], vars=[], values=[]):
        """
        Add new terms into left-hand-side of nonrenewable resource constraint.

            - Arguments:
                - coeffs: Coefficients for new terms; either a list of coefficients or a single coefficient.
                The three arguments must have the same size.
                - vars: Activity objects for new terms; either a list of activity objects or a single activity object.
                The three arguments must have the same size.
                - values: Mode objects for new terms; either a list of mode objects or a single mode object.
                The three arguments must have the same size.

            - Example usage:

            >>> budget.addTerms(1,act,express)

            adds one unit of nonrenewable resource (budget) if activity "act" is executed in mode "express."

        """

        if type(coeffs) != type([]):  # need a check whether coeffs is numeric ...
            self.terms.append((coeffs, vars, values))
        elif type(coeffs) != type([]) or type(vars) != type([]) or type(values) != type([]):
            print("coeffs, vars, values must be lists")
            raise TypeError
        elif len(coeffs) != len(vars) or len(coeffs) != len(values) or len(values) != len(vars):
            print("length of coeffs, vars, values must be identical")
            raise TypeError
        else:
            for i in range(len(coeffs)):
                self.terms.append((coeffs[i], vars[i], values[i]))

    def setRhs(self, rhs=0):
        """
        Sets the right-hand-side of linear constraint.
        
            - Argument:
                - rhs: Right-hand-side of linear constraint.
                
            - Example usage:
            
            >>> L.setRhs(10)
            
        """
        self.rhs = rhs

    def setDirection(self, direction="<="):
        if direction in ["<=", ">=", "="]:
            self.direction = direction
        else:
            print("direction setting error; direction should be one of '<=' or '>=' or '='")
            raise NameError


class Temporal():
    def __init__(self, pred, succ, tempType, delay):
        """
        OptSeq temporal class.

        A temporal constraint has the following form::
        
            predecessor's completion (start) time +delay <=
                            successor's start (completion) time.

        Parameter "delay" can be negative.  

            - Arguments:
                - pred: Predecessor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.   
                - succ: Successor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.  
                - tempType (optional): String that differentiates the temporal type.
                    "CS" (default)=Completion-Start, "SS"=Start-Start, 
                    "SC"= Start-Completion, "CC"=Completion-Completion.
                - delay (optional): Time lag between the completion (start) times of two activities.

            - Attributes:
                - pred: Predecessor (an activity object) or string "source."
                - succ: Successor (an activity object) or string "source."
                - type: String that differentiates the temporal type.
                    "CS" (default)=Completion-Start, "SS"=Start-Start, 
                    "SC"= Start-Completion, "CC"=Completion-Completion.
                - delay: Time lag between the completion (start) times of two activities.

        """

        self.pred = pred
        self.succ = succ
        self.type = tempType
        self.delay = delay

    def __str__(self):
        if self.pred == "source":
            pred = "source"
        elif self.pred == "sink":
            pred = "sink"
        else:
            pred = str(self.pred.name)

        if self.succ == "source":
            succ = "source"
        elif self.succ == "sink":
            succ = "sink"
        else:
            succ = str(self.succ.name)

        ret = ["temporal {0} {1}".format(pred, succ)]
        ret.append(" type {0} delay {1} ".format(self.type, self.delay))

        return " ".join(ret)


class State():
    ID = 0

    def __init__(self, name=""):
        """
        OptSeq state class.

        You can create a state object by adding a state to a model (using model.addState)
        instead of by using a State constructor.    

            - Arguments:
                - name: Name of state. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                   
        """
        if name == "" or name == None:
            name = "__s{0}".format(State.ID)
            State.ID += 1
        # convert illegal characters into _ (underscore)
        self.name = str(name).translate(_trans)

        self.Value = {}  # dictionary that maps time (non-negative integer) to value (non-negative integer)

    def __str__(self):
        ret = ["state {0} ".format(self.name)]
        for v in self.Value:
            ret.append("time {0} value {1} ".format(v, self.Value[v]))
        return " ".join(ret)

    def addValue(self, time=0, value=0):
        """
        Adds a value to the state
            - Arguments:
                - time: the time at which the state changes.
                - value: the value that the state changbes to
                
            - Example usage:

            >>> state.addValue(time=5,value=1)               
        """
        if type(time) == type(1) and type(value) == type(1):
            self.Value[time] = value
        else:
            print("time and value of the state {0} must be integer".format(self.name))
            raise TypeError


class Model(object):
    def __init__(self, name=""):
        """
        OptSeq model class.
            - Attributes:
                - activities: Dictionary that maps activity names to activity objects in the model.
                - modes: Dictionary that maps mode names to mode objects in the model.
                - resources:  Dictionary that maps resource names to resource objects in the model.
                - temporals: Dictionary that maps pairs of activity names to temporal constraint objects in the model.
                - Params: Object including all the parameters of the model.

                - act: List of all the activity objects in the model.
                - res: List of all the resource objects in the model.
                - tempo: List of all the tamporal constraint objects in the model.
        """
        self.name = name
        self.activities = {}  # set of activities maintained by a dictionary
        self.modes = {}  # set of modes maintained by a dictionary
        self.resources = {}  # set of resources maintained by a dictionary
        self.temporals = {}  # set of temporal constraints maintained by a dictionary
        self.states = {}  # set of states maintained by a dictionary

        self.act = []  # list of activity objects
        self.res = []  # list of resource objects
        self.tempo = []  # list of temporal constraint's objects
        self.state = []  # list of state objects

        self.Params = Parameters()  # controal parameters' class

    def __str__(self):
        ret = ["model:{0}".format(self.name)]
        ret.append("number of activities= {0}".format(len(self.act)))
        ret.append("number of resources= {0}".format(len(self.res)))

        if len(self.res):
            ret.append("\nResource Information")
            for res in self.res:
                ret.append(str(res))
                if len(res.terms) > 0:
                    ret.append(res.printConstraint())

        for a in self.act:
            if len(a.modes) >= 2:
                for m in a.modes:
                    self.modes[m.name] = m

        if len(self.modes):
            ret.append("\nMode Information")
            for i in self.modes:
                # ret.append("{0}\n{1}".format(i,self.modes[i]))
                ret.append(str(i))
                ret.append(str(self.modes[i]))

        if len(self.act):
            ret.append("\nActivity Information")
            for act in self.act:
                ret.append(str(act))

        if len(self.tempo):
            ret.append("\nTemporal Constraint Information")
            for t in self.tempo:
                ret.append(str(t))

        if len(self.state):
            ret.append("\nState Information")
            for s in self.state:
                ret.append(str(s))

        return "\n".join(ret)

    def addActivity(self, name="", duedate="inf", weight=1, autoselect=False):
        """
        Add an activity to the model.

            - Arguments:
                - name: Name for new activity. A string object except "source" and "sink." Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                - duedate(optional): Duedate of activity. A non-nagative integer or string "inf."
                - weight(optional): Panalty of one unit of tardiness. Positive integer.
                - autoselect(optional): True or False flag that indicates the activity selects the mode automatically or not.
 
            - Return value: New activity object.

            - Example usage:

            >>> a = model.addActivity("act1")
            
            >>> a = model.addActivity(name="act1",duedate=20,weight=100)
            
            >>> a = model.addActivity("act1",20,100)                    
        """
        activity = Activity(name, duedate, weight, autoselect)
        self.act.append(activity)
        # self.activities[activity.name]=activity
        return activity

    def addResource(self, name="", capacity={}, rhs=0, direction="<=", weight="inf"):
        """
        Add a resource to the model.

            - Arguments:
                - name: Name for new resource. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
                - capacity (optional): Capacity dictionary of the renewable (standard) resource.
                - Capacity dictionary maps intervals (pairs of start time and finish time) to amounts of capacity.
                - rhs (optional): Right-hand-side constant of nonrenewable resource constraint. 
                - direction (optional): Rirection (or sense) of nonrenewable resource constraint; "<=" (default) or ">=" or "=".
                - weight (optional): Weight of resource. Non-negative integer or "inf" (default).

            - Return value: New resource object.

            - Example usage:

            >>> r=model.addResource("res1")                             

            >>> r=model.addResource("res1", {(0,10):1,(12,100):2} )    

            >>> r=model.addResource("res2",rhs=10,direction=">=")      

        """
        res = Resource(name, capacity, rhs, direction, weight)
        self.res.append(res)
        # self.resources[res.name]=res
        return res

    def addTemporal(self, pred, succ, tempType="CS", delay=0):
        """
        Add a temporal constraint to the model.

        A temporal constraint has the following form::
        
            predecessor's completion (start) time +delay <=
                            successor's start (completion) time.

        Parameter "delay" can be negative.  

            - Arguments:
                - pred: Predecessor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.   
                - succ: Successor (an activity object) or string "source."
                        Here, "source" specifies a dummy activity that precedes all other activities and starts at time 0.   
                - tempType (optional): String that differentiates the temporal type.
                    "CS" (default)=Completion-Start, "SS"=Start-Start, 
                    "SC"= Start-Completion, "CC"=Completion-Completion.
                - delay (optional): Time lag between the completion (start) times of two activities.

            - Return value: New temporal object.

            - Example usage:

            >>> t=model.addTemporal(act1,act2)                         

            >>> t=model.addTemporal(act1,act2,type="SS",delay=-10)

            To specify the start time of activity act is exactly 50, we use two temporal constraints:

            >>> t=model.addTemporal("source",act,type="SS",delay=50)

            >>> t=model.addTemporal(act,"source",type="SS",delay=50)           
        """
        t = Temporal(pred, succ, tempType, delay)
        self.tempo.append(t)
        # self.temporals[pred.name,succ.name]=None
        return t

    def addState(self, name=""):
        """
        Add a state to the model.

            - Arguments:
                - name: Name for new state. Remark that strings in OptSeq are restricted to a-z, A-Z, 0-9,[],_ and @.
 
            - Return value: New state object.

            - Example usage:

            >>> a = model.addState("state1")
                              
        """
        s = State(name)
        self.state.append(s)
        # self.states[name]=s
        return s

    def update(self):
        """
        prepare a string representing the current model in the OptSeq input format
        """
        makespan = self.Params.Makespan

        f = []

        self.resources = {}  # dictionary of resources that maps res-name to res-object
        for r in self.res:
            self.resources[r.name] = r
            f.append(str(r))

        self.states = {}  # dictionary of activities that maps state-name to state-object
        for s in self.state:
            self.states[s.name] = s
            f.append(str(s))

        self.modes = {}  # dictionary of modes that maps mode-name to mode-object
        for a in self.act:
            if len(a.modes) >= 2:
                for m in a.modes:
                    self.modes[m.name] = m

        for m in self.modes:  # print mode information
            f.append("mode {0} ".format(m))
            f.append(str(self.modes[m]))

        self.activities = {}  # dictionary of activities that maps activity-name to activity-object
        for a in self.act:
            self.activities[a.name] = a
            f.append(str(a))

        self.temporals = {}  # dictionary of temporal constraints that maps activity name pair to temporal-object
        for t in self.tempo:
            if t.pred == "source":
                pred = "source"
            elif t.pred == "sink":
                pred = "sink"
            else:
                pred = t.pred.name

            if t.succ == "source":
                succ = "source"
            elif t.succ == "sink":
                succ = "sink"
            else:
                succ = t.succ.name

            self.temporals[(pred, succ)] = t
            f.append(str(t))

        # non-renewable constraint
        for r in self.res:
            self.resources[r.name] = r
            if len(r.terms) > 0:
                f.append(r.printConstraint())

        if makespan:
            f.append("activity sink duedate 0 \n")
        return " \n".join(f)

    def optimize(self):
        """
        Optimize the model using optseq.exe in the same directory.

            - Example usage:

            >>> model.optimize()
        """
        LOG = self.Params.OutputFlag

        """
        SCOP input file
        """
        f = self.update()
        f2 = open("optseq_input.txt", "w")
        f2.write(f)
        f2.close()

        import subprocess
        cmd = optseq + " -time " + str(self.Params.TimeLimit) + \
              " -backtrack  " + str(self.Params.Backtruck) + \
              " -iteration  " + str(self.Params.MaxIteration) + \
              " -report     " + str(self.Params.ReportInterval) + \
              " -seed      " + str(self.Params.RandomSeed) + \
              " -tenure    " + str(self.Params.Tenure)

        if self.Params.Initial:
            cmd += " -initial optseq_best_act_data.txt"

        # print "cmd=",cmd

        try:
            # pipe = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
            #                        stdin=subprocess.PIPE, shell=True)
            pipe = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                                    stdin=subprocess.PIPE, shell=True)
            print("\n ================ Now solving the problem ================ \n")
        except OSError:
            print("error: could not execute command '%s'" % cmd)
            print("please check that the solver is in the path")
            exit(0)

        out, err = pipe.communicate(f.encode())  # get the result

        if err != None:
            if int(sys.version_info[0]) >= 3:
                err = str(err, encoding='utf-8')
            f2 = open("optseq_error.txt", "w")
            f2.write(err)
            f2.close()

        if int(sys.version_info[0]) >= 3:
            out = str(out, encoding='utf-8')

        if LOG:
            print("\noutput:")
            print(out)

        print("\nSolutions:")

        """
        optseq output file
        """
        f3 = open("optseq_output.txt", "w")
        f3.write(out)
        f3.close()

        # search strings
        s0 = "--- best solution ---"
        s1 = "--- tardy activity ---"
        s2 = "--- resource residuals ---"
        s3 = "--- best activity list ---"  # added for optseq 3.0
        s4 = "objective value ="
        pos0 = out.find(s0) + len(s0)  # job data start position
        pos1 = out.find(s1, pos0)  # data end position
        pos2 = out.find(s2, pos1)
        pos3 = out.find(s3, pos2)
        pos4 = out.find(s4, pos3)
        # print("data positions",pos0,pos1,pos2,pos3,pos4)
        data = out[pos0:pos1]
        resdata = out[pos2 + len(s2):pos3]
        data = data.splitlines()
        reslines = resdata.splitlines()

        # save the best activity list
        bestactdata = out[pos3 + len(s3):pos4]
        f3 = open("optseq_best_act_data.txt", "w")
        f3.write(bestactdata.lstrip())
        f3.close()

        for line in reslines:
            if len(line) <= 1:
                continue
            current = line.split()
            resname = current[0][:-1]
            residual = current[1:]
            count = 0
            resDic = {}  # residual capacity
            while count < len(residual):
                interval = residual[count].split(",")
                int1 = int(interval[0][1:])
                int2 = int(interval[1][:-1])
                count += 1
                num = int(residual[count])
                count += 1
                resDic[(int1, int2)] = num
            # print(resname,residual)
            self.resources[resname].residual = resDic

        # job data conversion
        execute = []
        for i in range(len(data)):
            replaced = data[i].replace(",", " ")
            current = replaced.split()  # split by space
            # print(current)
            if len(current) > 1:
                execute.append(current)
        for line in execute:
            # print("line=",line)
            actname = line[0]
            mode = line[1]
            try:
                start = line[2]
            except:
                print("Problem is infeasible")
                exit(0)

            execute = line[3:-1]  # list for breakable activity
            completion = line[-1]
            print("{0:>10} {1:>5} {2:>5} {3:>5}".format(actname, mode, start, completion))
            # print("execute=",execute)
            if actname == "source":
                pass
            elif actname == "sink":
                pass
            else:
                self.activities[actname].start = int(start)
                self.activities[actname].completion = int(completion)
                if mode != "---":
                    self.activities[actname].selected = mode
                exeDic = {}
                for exe in execute:
                    exedata = exe.split("--")
                    start = exedata[0]
                    completion = exedata[1]
                    idx = completion.find("[")
                    # for parallel execution
                    if idx > 0:
                        parallel = completion[idx + 1:-1]
                        completion = completion[:idx]
                        # print(completion,idx,parallel)
                    else:
                        parallel = 1
                    exeDic[(int(start), int(completion))] = parallel
                self.activities[actname].execute = exeDic
        return

    def write(self, filename="optseq_chart.txt"):
        """
        Output the gantt's chart as a text file.

            - Argument:
                - filename: Output file name. Default="optseq_chart.txt."

            - Example usage:

            >>> model.write("sample.txt")

        """
        f = open(filename, "w")

        horizon = 0
        actList = []
        for a in self.activities:
            actList.append(a)
            act = self.activities[a]
            horizon = max(act.completion, horizon)
        # print("planning horizon=",horizon)
        actList.sort()
        title = " activity    mode".center(20) + " duration |"

        width = len(str(horizon))  # period width =largest index of time
        for t in range(horizon):
            num = str(t + 1)
            title += num.rjust(width) + "|"
            # print(title)
        f.write(title + "\n")
        f.write("-" * (30 + (width + 1) * horizon) + "|\n")
        for a in actList:  # sorted order
            act = self.activities[a]  # act: activity object
            actstring = act.name.center(10)[:10]
            if len(act.modes) >= 2:
                actstring += str(act.selected).center(10)
                actstring += str(self.modes[act.selected].duration).center(10)
                # print(" executed on resource:")
                # print(self.modes[act.selected].requirement,m1.modes[act.selected].rtype)
            else:
                # print("executed on resource:")
                # print(act.modes[0].requirement,act.modes[0].rtype)
                actstring += str(act.modes[0].name).center(10)[:10]
                actstring += str(act.modes[0].duration).center(10)
            execute = [0 for t in range(horizon)]
            for (s, c) in act.execute:
                para = act.execute[s, c]
                for t in range(s, c):
                    execute[t] = int(para)

            for t in range(horizon):
                if execute[t] >= 2:
                    # for res_name in self.modes[act.selected].requirement:
                    # print(res_name)
                    # print(m1.modes[act.selected].rtype)
                    # print(self.modes[act.selected])
                    actstring += "|*" + str(execute[t]).rjust(width - 1)
                elif execute[t] == 1:
                    actstring += "|" + "=" * (width)
                elif t >= act.start and t < act.completion:
                    actstring += "|" + "." * (width)
                else:
                    actstring += "|" + " " * width
            actstring += "|"
            # print(actstring)
            f.write(actstring + "\n")
        ##    ##        print(act.name +"  starts at "+str(act.start)+" and finish at " +str(act.completion))
        ##    ##        print("  and is executed :"+str(act.execute)])

        f.write("-" * (30 + (width + 1) * horizon) + "\n")
        f.write("resource usage/capacity".center(30) + "| \n")
        f.write("-" * (30 + (width + 1) * horizon) + "\n")
        resList = []
        for r in self.resources:
            resList.append(r)
        resList.sort()
        for r in resList:
            res = self.resources[r]
            if len(res.terms) == 0:  # output residual and capacity
                rstring = res.name.center(30)
                cap = [0 for t in range(horizon)]
                residual = [0 for t in range(horizon)]
                for (s, c) in res.residual:
                    amount = res.residual[(s, c)]
                    if c == "inf":
                        c = horizon
                    s = min(s, horizon)
                    c = min(c, horizon)
                    for t in range(s, c):
                        residual[t] += amount

                for (s, c) in res.capacity:
                    amount = res.capacity[(s, c)]
                    if c == "inf":
                        c = horizon
                    s = min(s, horizon)
                    c = min(c, horizon)
                    for t in range(s, c):
                        cap[t] += amount

                for t in range(horizon):
                    num = str(cap[t] - residual[t])
                    rstring += "|" + num.rjust(width)
                f.write(rstring + "|\n")

                rstring = str(" ").center(30)

                for t in range(horizon):
                    num = str(cap[t])
                    rstring += "|" + num.rjust(width)
                f.write(rstring + "|\n")
                f.write("-" * (30 + (width + 1) * horizon) + "\n")
        f.close()

    def writeExcel(self, filename="optseq_chart.csv", scale=1):
        """
        Output the gantt's chart as a csv file for printing using Excel.

            - Argument:
                - filename: Output file name. Default="optseq_chart.csv."

            - Example usage:

            >>> model.writeExcel("sample.csv")

        """
        f = open(filename, "w")
        horizon = 0
        actList = []
        for a in self.activities:
            actList.append(a)
            act = self.activities[a]
            horizon = max(act.completion, horizon)
        # print("planning horizon=",horizon)
        if scale <= 0:
            print("optseq write scale error")
            exit(0)
        original_horizon = horizon
        horizon = int(horizon / scale) + 1
        actList.sort()
        title = " activity ,   mode,".center(20) + " duration,"
        width = len(str(horizon))  # period width =largest index of time
        for t in range(horizon):
            num = str(t + 1)
            title += num.rjust(width) + ","
        f.write(title + "\n")
        for a in actList:  # sorted order
            act = self.activities[a]  # act: activity object
            actstring = act.name.center(10)[:10] + ","
            if len(act.modes) >= 2:
                actstring += str(act.selected).center(10) + ","
                actstring += str(self.modes[act.selected].duration).center(10) + ","
            else:
                actstring += str(act.modes[0].name).center(10)[:10] + ","
                actstring += str(act.modes[0].duration).center(10) + ","
            execute = [0 for t in range(horizon)]
            for (s, c) in act.execute:
                para = act.execute[s, c]
                for t in range(s, c):
                    t2 = int(t / scale)
                    execute[t2] = int(para)
            for t in range(horizon):
                if execute[t] >= 2:
                    actstring += "*" + str(execute[t]).rjust(width - 1) + ","
                elif execute[t] == 1:
                    actstring += "" + "=" * (width) + ","
                elif t >= int(act.start / scale) and t < int(act.completion / scale):
                    actstring += "" + "." * (width) + ","
                else:
                    actstring += "" + " " * width + ","
            f.write(actstring + "\n")
        resList = []
        for r in self.resources:
            resList.append(r)
        resList.sort()

        for r in resList:
            res = self.resources[r]
            if len(res.terms) == 0:  # output residual and capacity
                rstring = res.name.center(30) + ", , ,"
                cap = [0 for t in range(horizon)]
                residual = [0 for t in range(horizon)]
                for (s, c) in res.residual:
                    amount = res.residual[(s, c)]
                    if c == "inf":
                        c = horizon
                    s = min(s, original_horizon)
                    c = min(c, original_horizon)
                    s2 = int(s / scale)
                    c2 = int(c / scale)
                    for t in range(s2, c2):
                        residual[t] += amount

                for (s, c) in res.capacity:
                    amount = res.capacity[(s, c)]
                    if c == "inf":
                        c = horizon
                    s = min(s, original_horizon)
                    c = min(c, original_horizon)
                    s2 = int(s / scale)
                    c2 = int(c / scale)
                    for t in range(s2, c2):
                        cap[t] += amount

                for t in range(horizon):
                    # num=str(cap[t]-residual[t])
                    rstring += str(residual[t]) + ","
                f.write(rstring + "\n")

                # rstring= str(" ").center(30)+", , ,"
                #        
                # for t in range(horizon):
                #    num=str(cap[t])
                #    rstring+=""+num.rjust(width) +","
                # f.write(rstring+"\n")
        f.close()

    # if __name__=="__main__":

##    m1=model()
##    #resource declaration  
##    machine={} #define three machines
##    for j in range(1,4):
##        machine[j]=m1.addResource("machine[%s]"%j,capacity={(0,"inf"):1})
##
##    #CAP={} #capacity of human resources; two workers are available on weekdays
##    #for t in range(9):
##    #    CAP[(t*7,t*7+5)]=2
##    #manpower=m1.addResource("manpower",capacity=CAP)
##    # we may write ...
##    manpower=m1.addResource("manpower")
##    for t in range(9):
##        manpower.addCapacity(t*7,t*7+5,2)
##
##    #budget constraint
##    budget=m1.addResource(name="budget_constraint",rhs=1,direction="=")
##
##    #activity declaration (3 activities are processed on three machines)
##    #JobInfo containes the info. of operations (activity,1..3):-> machine ID and proc. time 
##    JobInfo={ (1,1):(1,7), (1,2):(2,10), (1,3):(3,4),
##              (2,1):(3,9), (2,2):(1,5), (2,3):(2,11),
##              (3,1):(1,3), (3,2):(3,9), (3,3):(2,12),
##              (4,1):(2,6), (4,2):(3,13), (4,3):(1,9)
##        }
##    act={}
##    express=Mode("Express",duration=4)
##    express.addResource(machine[2],{(0,"inf"):1},"max")
##    express.addResource(manpower,{(0,2):1})
##    express.addBreak(1,1)
##    #express.addParallel(1,1,2)
##        
##    mode={}
##    for (i,j) in JobInfo: #for each job and machine
##        act[i,j]=m1.addActivity("Act[%s][%s]"%(i,j))
##        mode[i,j]=Mode("Mode[%s][%s]"%(i,j),duration=JobInfo[i,j][1])
##        mode[i,j].addResource(machine[JobInfo[i,j][0]],{(0,"inf"):1},"max")
##        mode[i,j].addResource(manpower,{(0,2):1})
##        mode[i,j].addBreak(1,1)
##        if JobInfo[i,j][0]==1:
##            mode[i,j].addParallel(1,1,2)
##
##        
##        if JobInfo[i,j][0]==2:
##            #activities processed on machine 2 have two modes, Express and Normal.
##            act[i,j].addModes(mode[i,j],express)
##            #Express mode needs one budget
##            budget.addTerms(1,act[i,j],express)
##        else:
##            act[i,j].addModes(mode[i,j]) #single mode activity
##    #temporal (precedense) constraints 
##    for i in range(1,5):
##        for j in range(1,3):
##            m1.addTemporal(act[i,j],act[i,j+1])   
##
##    #Define that Act[2][2] must be processed just after Act[1][1] on machine 1
##    #introduce dummy with duration 0 and can break at time 0
##    #it requires machine 1 during the break,
##    #  completion of Act[1][1]=start of dummy and completaion of dummy=start of Act[2][2]  
##    d_act=m1.addActivity("dummy_activity")
##    d_mode=Mode("dummy_mode")
##    d_mode.addBreak(0,0)
##    d_mode.addResource(machine[1],{(0,0):1},"break")
##    d_act.addModes(d_mode)
##    m1.addTemporal(act[1,1],d_act,tempType="CS")
##    m1.addTemporal(d_act,act[1,1],tempType="SC")
##    m1.addTemporal(d_act,act[2,2],tempType="CS")
##    m1.addTemporal(act[2,2],d_act,tempType="SC")
##    
##    #m1.addTemporal("sink","source",tempType="CS",delay=-10)
##    
##    print(m1)
##    m1.Params.TimeLimit=1
##    m1.Params.OutputFlag=True
##    m1.Params.Makespan=True
##    m1.optimize()
##    m1.write("chart11.txt")

##    m1=model()
##
##    duration={(1,1):3,(1,2):3,(1,3):3,(2,1):3,(2,2):3,(2,3):3}
##    setup={(1,1):1,(1,2):5,(2,1):5,(2,2):1}
##
##    act={}
##    act_setup={}
##    mode={}
##    mode_setup={}
##    rs=m1.addResource("machine",1)
##    s1=m1.addState("Setup_State")
##    s1.addValue(time=0,value=1)
##
##    s2=m1.addState("Setup_State2")
##    s2.addValue(time=1,value=2)
##
##    #setup mode
##    for (i,j) in setup:
##        mode_setup[i,j]=Mode("Mode_setup"+str(i)+str(j),setup[i,j])
##        mode_setup[i,j].addState(s1,i,j)
##        mode_setup[i,j].addResource(rs,{(0,"inf"):1})
##
##    for (i,j) in duration:
##        act_setup[i,j]=m1.addActivity("Setup"+str(i)+str(j),autoselect=True)
##        act_setup[i,j].addModes(mode_setup[1,i],mode_setup[2,i])
##        act[i,j]=m1.addActivity("Act"+str(i)+str(j))
##        mode[i,j]=Mode("Mode"+str(i)+str(j),duration[i,j])
##        mode[i,j].addResource(rs,{(0,"inf"):1})
##        mode[i,j].addBreak(0,0)
##        mode[i,j].addResource(rs,{(0,"inf"):1},"break")
##        act[i,j].addModes(mode[i,j])
##
##    #temporal (precedense) constraints
##    for (i,j) in duration:
##        m1.addTemporal(act_setup[i,j],act[i,j],"CS")
##        m1.addTemporal(act[i,j],act_setup[i,j],"SC")
##
##    print(m1)  
##    m1.Params.TimeLimit=1
##    m1.Params.OutputFlag=0
##    m1.Params.Makespan=True
##    m1.optimize()
##    ###m1.write("chart1.txt")    

#     m1=model()
# 
#     due={1:5,2:9,3:6,4:4}
#     duration={1:1, 2:2, 3:3, 4:4 }
# 
#     res=m1.addResource("writer")
#     res.addCapacity(0,3,1)
#     res.addCapacity(4,6,1)
#     res.addCapacity(7,10,1)
#     res.addCapacity(11,"inf",1)
# 
#     act={}
#     mode={}
# 
#     for i in duration: 
#         act[i]=m1.addActivity("Act[{0}]".format(i),duedate=due[i])
#         mode[i]=Mode("Mode[{0}]".format(i),duration[i])
#         mode[i].addResource(res,{(0,"inf"):1})
#         mode[i].addBreak(0,"inf",1)
#         # mode[i].addResource(res,{(0,"inf"):1},"break")       
#         act[i].addModes(mode[i])
#         
#     print (m1)
# 
#     
#     m1.Params.TimeLimit=1
#     m1.Params.OutputFlag=True
#     m1.Params.Makespan=False
#     m1.optimize()
#     m1.write("chart9.txt")
