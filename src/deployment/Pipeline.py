from dao.DataIO import DataIO
from common.SqlConfig import SqlConfig
from init.DataLoad import DataLoad


class Pipeline(object):
    def __init__(self):
        self.io = DataIO
        self.sql_conf = SqlConfig

        # Factory Information
        self.mst = {}

    def run(self):
        # Initialize load class
        load_cls = DataLoad(io=self.io, sql_conf=self.sql_conf)

        # Load dataset
        mst = load_cls.load_mst_temp()    # Master dataset
        demand = load_cls.load_demand_temp()