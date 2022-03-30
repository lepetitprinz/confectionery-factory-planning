from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from init.DataLoadDev import DataLoadDev
from init.PreprocessDev import PreprocessDev
from Model.OptSeqModel import OptSeqModel


class PipelineDev(object):
    def __init__(self):
        self.io = DataIO()
        self.sql_conf = SqlConfig()

        # Factory Information
        self.dmd_plant_list = []
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

        # Demand
        self.dmd_plant_list, dmd_by_plant = prep.set_dmd_info(data=demand)    # Demand

        # Resource
        res_grp_by_plant = prep.set_res_grp(data=mst['resource'])    # Resource
        res_grp_item_by_plant = prep.set_res_grp_item(data=mst['res_grp_item'])

        # Bom route
        bom_by_plant = prep.set_bom_route_info(data=mst['bom_route'])
        # oper_by_plant = prep.set_oper_info(data=mst['operation'])

        # Model
        for plant in self.dmd_plant_list:
            opt_seq = OptSeqModel(
                item_res_grp=None,
                res_grp=res_grp_by_plant[plant],
                res_grp_item=res_grp_item_by_plant[plant],
                res_grp_duration={}
            )

            # Initialize model
            model = opt_seq.init(
                dmd_list=dmd_by_plant[plant]
            )
            # plan.after_process(operation=operation)
