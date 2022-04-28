import datetime

import common.util as util

import datetime as dt


class Init(object):
    def __init__(self, io, sql_conf, default_path: dict, fp_num: str, fp_seq: str):
        # io class instance attribute
        self.io = io
        self.sql_conf = sql_conf
        self.default_path = default_path
        self.pipeline_path = {}

        # Forward planning instance attribute
        self.fp_num = fp_num
        self.fp_seq = fp_seq
        self.fp_version = ''
        # self.fp_version = 'FP_2022W16.01'    # Todo : Temp

        # Time instance attribute
        self.calendar = None
        self.plant_start_hour = 7
        self.plant_start_day = None

    def run(self):
        self.set_calendar()
        self.set_fp_version()
        self.set_pipeline_path()
        self.set_plant_start_time()

    def set_calendar(self) -> None:
        self.calendar = self.io.load_from_db(sql=self.sql_conf.sql_calendar())

    def set_fp_version(self) -> None:
        today = dt.date.today().strftime('%Y%m%d')

        # Set forward planning version
        today_df = self.calendar[self.calendar['yymmdd'] == today]
        year = today_df['yy'].values[0]
        week = today_df['week'].values[0]
        self.fp_version = util.make_fp_version_name(year=year, week=week, seq=self.fp_num)

    def set_plant_start_time(self) -> None:
        today = dt.datetime.combine(dt.datetime.today(), dt.datetime.min.time())
        today = today - datetime.timedelta(days=today.weekday())    # This monday

        # self.plant_start_day = today + dt.timedelta(hours=self.plant_start_hour)
        self.plant_start_day = today

    def set_pipeline_path(self) -> None:
        self.pipeline_path = {
            'load_data': util.make_vrsn_path(
                path=self.default_path['save'],
                module='load',
                version=self.fp_version,
                name='data' + self.fp_seq,
                extension='pickle'
            ),
            'load_master': util.make_vrsn_path(
                path=self.default_path['save'],
                module='load',
                version=self.fp_version,
                name='master_' + self.fp_seq,
                extension='pickle'
            ),
            'load_demand': util.make_vrsn_path(
                path=self.default_path['save'],
                module='load',
                version=self.fp_version,
                name='demand_' + self.fp_seq,
                extension='pickle'
            ),
            'prep_data': util.make_vrsn_path(
                path=self.default_path['save'],
                module='prep',
                version=self.fp_version,
                name='data_' + self.fp_seq,
                extension='pickle'
            ),
            'model': util.make_vrsn_path(
                path=self.default_path['save'],
                module='model',
                version=self.fp_version,
                name='model_' + self.fp_seq,
                extension='pickle'
            ),
        }
