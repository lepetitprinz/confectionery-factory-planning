from common.SqlSession import SqlSession
from common.SqlConfig import SqlConfig

import json
import pickle
import pandas as pd


class DataIO(object):
    def __init__(self):
        self.sql_conf = SqlConfig()
        self.session = SqlSession()
        self.session.init()

    # Read sql and converted to dataframe
    def get_df_from_db(self, sql, dtype=None) -> pd.DataFrame:
        df = self.session.select(sql=sql, dtype=dtype)
        df.columns = [col.lower() for col in df.columns]

        return df

    # Read sql and converted to dictionary
    def get_dict_from_db(self, sql, key, val, dtype=None) -> dict:
        df = self.session.select(sql=sql, dtype=dtype)
        df[key] = df[key].apply(str.lower)
        result = df.set_index(keys=key).to_dict()[val]

        return result

    # Insert dataframe on DB
    def insert_to_db(self, df: pd.DataFrame, tb_name: str, verbose=True) -> None:
        self.session.insert(df=df, tb_name=tb_name, verbose=verbose)

    # Delete from DB
    def delete_from_db(self, sql: str) -> None:
        self.session.delete(sql=sql)

    # Update on DB
    def update_from_db(self, sql: str) -> None:
        self.session.update(sql=sql)

    # Save the object
    @staticmethod
    def save_object(data, data_type: str, file_path: str) -> None:
        """
        :param data: Saving dataset
        :param data_type: csv / binary
        :param file_path: file path
        """
        if data_type == 'csv':
            data.to_csv(file_path, index=False)

        elif data_type == 'binary':
            with open(file_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                handle.close()

        elif data_type == 'json':
            with open(file_path, 'w') as handle:
                json.dump(data, handle, indent=4)

    # Load the object
    @staticmethod
    def load_object(file_path: str, data_type: str):
        data = None
        if data_type == 'csv':
            data = pd.read_csv(file_path)
            # data = pd.read_csv(file_path, encoding='cp949')

        elif data_type == 'binary':
            with open(file_path, 'rb') as handle:
                data = pickle.load(handle)

        elif data_type == 'json':
            with open(file_path, 'r') as handle:
                data = json.load(handle)

        return data
