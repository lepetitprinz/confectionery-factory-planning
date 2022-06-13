from common.session import Session
from common.sql import Query

import json
import pickle
import pandas as pd


class DataIO(object):
    def __init__(self):
        self.sql_conf = Query()
        self.session = Session()
        self.session.init()

    # Read sql and converted to dataframe
    def load_from_db(self, sql, dtype=None) -> pd.DataFrame:
        df = self.session.select(sql=sql, dtype=dtype)
        df.columns = [col.lower() for col in df.columns]

        return df

    # Insert dataframe on DB
    def insert_to_db(self, df: pd.DataFrame, tb_name: str, verbose=True) -> None:
        self.session.insert(df=df, tb_name=tb_name, verbose=verbose)

    # Delete from DB
    def delete_from_db(self, sql: str) -> None:
        self.session.delete(sql=sql)

    # Save the object
    @staticmethod
    def save_object(data, data_type: str, path: str) -> None:
        if data_type == 'csv':
            data.to_csv(path, index=False)

        elif data_type == 'binary':
            with open(path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()

        elif data_type == 'json':
            with open(path, 'w') as handle:
                json.dump(data, handle, indent=4)

    # Load the object
    @staticmethod
    def load_object(path: str, data_type: str):
        data = None
        if data_type == 'csv':
            data = pd.read_csv(path)

        elif data_type == 'binary':
            with open(path, 'rb') as handle:
                data = pickle.load(handle)

        elif data_type == 'json':
            with open(path, 'r') as handle:
                data = json.load(handle)

        return data
