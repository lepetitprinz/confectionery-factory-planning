from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from init.DataLoad import DataLoad
from init.Preprocessing import Preprocessing
from plan.Plan import Plan


class Pipeline(object):
    def __init__(self):
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Factory Information
        self.mst = {}

    def run(self):
        # Initialize load class
        load_cls = DataLoad(io=self.io, sql_conf=self.sql_conf)

        # Load dataset
        # mst = load_cls.load_mst_temp()    # Master dataset
        # demand = load_cls.load_demand_temp()

        mst = load_cls.load_mst()    # Master dataset
        demand = load_cls.load_demand()

        # Data preprocessing
        prep = Preprocessing()
        mst_map, demand, dmd_qty, bom_route, operation = prep.run(mst=mst, demand=demand)

        # Model
        plan = Plan(mst=mst, mst_map=mst_map, demand=demand)
        model = plan.init(
            dmd_qty=dmd_qty,
            bom_route=bom_route,
            operation=operation
        )
        plan.run(model=model)
        # plan.after_process(operation=operation)
