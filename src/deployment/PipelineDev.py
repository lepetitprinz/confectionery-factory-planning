from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from init.DataLoadDev import DataLoadDev
from init.PreprocessDev import PreprocessDev
from plan.Plan import Plan


class PipelineDev(object):
    def __init__(self):
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Factory Information
        self.dmd_fac_list = []
        self.mapping_info = {}
        self.mst = {}

    def run(self):
        # Initialize load class
        load = DataLoadDev(io=self.io, sql_conf=self.sql_conf)

        # Load dataset
        demand = load.load_demand()
        mst = load.load_info()    # Master dataset

        # Data preprocessing
        prep = PreprocessDev()

        self.dmd_fac_list, dmd_by_plant = prep.set_dmd_info(data=demand)    # Demand
        res_cnt_capa_by_plant = prep.set_res_cnt_capa(data=mst['res_cnt_capa'])    # Resource
        res_item_by_plant = prep.set_res_item(data=mst['res_item'])
        bom_by_plant = prep.set_bom_route_info(data=mst['bom_route'])
        oper_by_plant = prep.set_oper_info(data=mst['operation'])

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
