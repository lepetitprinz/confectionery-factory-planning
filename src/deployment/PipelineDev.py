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
        self.dmd_plant_list, dmd_by_plant, dmd_due_date = prep.set_dmd_info(data=demand)    # Demand

        # Resource
        res_grp_by_plant = prep.set_res_grp(data=mst['res_grp'])    # Resource
        item_res_grp_by_plant = prep.set_item_res_grp(data=mst['item_res_grp'])
        item_res_grp_duration_by_plant = prep.set_item_res_duration(data=mst['item_res_duration'])

        # Bom route
        bom_by_plant = prep.set_bom_route_info(data=mst['bom_route'])

        # Model
        for plant in self.dmd_plant_list:
            opt_seq = OptSeqModel(
                dmd_due_date=dmd_due_date[plant],
                item_res_grp=item_res_grp_by_plant[plant],
                item_res_grp_duration=item_res_grp_duration_by_plant[plant],
            )

            # Initialize model
            model = opt_seq.init(
                dmd_list=dmd_by_plant[plant],
                res_grp_list=res_grp_by_plant[plant]

            )
            opt_seq.optimize(model=model)
            # plan.after_process(operation=operation)
