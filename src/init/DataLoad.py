import os


class DataLoad(object):
    def __init__(self, io, sql_conf):
        """
        :param io: Pipeline step configuration
        :param sql_conf: SQL configuration
        """
        self.io = io
        self.sql_conf = sql_conf
        self.base_dir = os.path.join('..', '..')

    def load_mst_temp(self):
        bom = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'bom.csv'), data_type='csv')
        item = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'item.csv'), data_type='csv')
        oper = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'operation.csv'), data_type='csv')
        res = self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'wc.csv'), data_type='csv')

        # change upper case to lower
        bom.columns = [col.lower() for col in bom.columns]
        item.columns = [col.lower() for col in item.columns]
        oper.columns = [col.lower() for col in oper.columns]
        res.columns = [col.lower() for col in res.columns]

        mst = {
            'bom': bom,
            'item': item,
            'oper': oper,
            'res': res
        }

        return mst

    def load_demand_temp(self):
        self.io.load_object(file_path=os.path.join(self.base_dir, 'data', 'demand.csv'), data_type='csv')
