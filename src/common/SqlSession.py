import common.config as config

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import Table, MetaData, insert
from sqlalchemy.sql import text
from sqlalchemy.exc import SQLAlchemyError


class SqlSession(object):
    def __init__(self):
        # Create Engine Format(Microsoft SQL Server)
        # self.url = config.RDMS + '://' + config.USER + ':' + config.PASSWORD + \
        #            '@' + config.HOST + ':' + config.PORT + '/' + config.DATABASE

        self.url = config.RDMS + '://' + config.USER + ':' + config.PASSWORD + \
                   '@' + config.HOST + ':' + config.PORT + '/' + config.DATABASE + '?charset=utf8'

        self.engine = None
        self._connection = None
        self._data = None

    def init(self):
        """
        Initialize Data Source, Connection object
        """
        self.engine = self.create_engine()
        self._connection = self.get_connection()
        # print("DB Connected\n")

    def set_data(self, data_source):
        self._data = data_source

    def create_engine(self):
        return create_engine(self.url)
        # return create_engine(self.url, fast_executemany=True)

    def get_connection(self):
        """
        return Data Source Connection
        """
        return self.engine.connect()
        # return self.engine.connect().execution_options(stream_results=True)

    def commit(self):
        """
        commit
        """
        return self._connection.commit()

    def rollback(self):
        """
        rollback
        """
        return self._connection.rollback()

    def close(self):
        """
        close session
        """
        try:
            self._connection.close()
            print("DB Session is closed\n")

        except SQLAlchemyError as e:
            error = str(e.__dict__['orig'])
            print(f"Error: {error}")

    def select(self, sql: str, dtype=None):
        """
        get query
        """
        if self._connection is None:
            raise ConnectionError('Session is not initialized\n')

        try:
            # data = pd.read_sql_query(sql, self._connection)

            # Memory handling
            dfs = []
            for chunk in pd.read_sql_query(sql, self._connection, chunksize=10000, dtype=dtype):
                dfs.append(chunk)
            data = pd.concat(dfs)

            return data

        except SQLAlchemyError as e:
            error = str(e)
            return error

    def insert(self, df: pd.DataFrame, tb_name: str, verbose: bool):
        # Get meta information
        table = self.get_table_meta(tb_name=tb_name)
        df.columns = [col.upper() for col in df.columns]

        with self.engine.connect() as conn:
            conn.execute(table.insert(), df.to_dict('records'))
            if verbose:
                print(f"Saving {tb_name} table is finished.\n")

    def delete(self, sql: str):
        statement = text(sql)
        with self.engine.connect() as conn:
            conn.execute(statement)

    def update(self, sql: str):
        statement = text(sql)
        with self.engine.connect() as conn:
            conn.execute(statement)

    def get_table_meta(self, tb_name: str):
        metadata = MetaData(bind=self.engine)
        metadata.reflect(self.engine, only=[tb_name])
        table = Table(tb_name, metadata, autoload=True, autoload_with=self.engine)

        return table