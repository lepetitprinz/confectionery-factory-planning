import os


class DataLoad(object):
    def __init__(self, io, sql_conf):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        """
        self.io = io
        self.sql_conf = sql_conf

    def load_mst_temp(self):
        bom = self.io.load_object(file_path=os.path.join('..', 'data', 'bom.csv'), data_type='csv')
        item = self.io.load_object(file_path=os.path.join('..', 'data', 'item.csv'), data_type='csv')
        oper = self.io.load_object(file_path=os.path.join('..', 'data', 'operation.csv'), data_type='csv')
        wc = self.io.load_object(file_path=os.path.join('..', 'data', 'wc.csv'), data_type='csv')

        mst = {
            'bom': bom,
            'item': item,
            'oper': oper,
            'wc': wc
        }

        return mst

    def load_demand_temp(self):
        self.io.load_object(file_path=os.path.join('..', 'data', 'demand.csv'), data_type='csv')
