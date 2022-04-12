import os
import common.util as util

import datetime as dt


class Init(object):
    def __init__(self, io, sql_conf, default_path: dict):
        self.io = io
        self.sql_conf = sql_conf
        self.default_path = default_path
        self.pipeline_path = {}

        self.calendar = None
        self.fp_version = ''
        self.plant_start_time = None

    def run(self):
        self.set_calendar()
        self.set_fp_version()
        self.set_pipeline_path()
        self.set_plant_start_time()

    def set_plant_start_time(self):
        self.plant_start_time = dt.datetime.today()

    def set_calendar(self):
        self.calendar = self.io.get_df_from_db(sql=self.sql_conf.sql_calendar())

    def set_fp_version(self):
        today = dt.date.today().strftime('%Y%m%d')

        # Set FP version
        today_df = self.calendar[self.calendar['yymmdd'] == today]
        year = today_df['yy'].values[0]
        week = today_df['week'].values[0]
        self.fp_version = year + week

    def set_pipeline_path(self):
        self.pipeline_path = {
            'load_master': util.make_path(
                path=self.default_path['save'],
                module='load',
                name='master',
                extension='pickle'
            ),
            'load_demand': util.make_version_path(
                path=self.default_path['save'],
                module='load',
                name='demand',
                version=self.fp_version,
                extension='pickle'
            ),
            'prep_demand': util.make_version_path(
                path=self.default_path['save'],
                module='prep',
                name='demand',
                version=self.fp_version,
                extension='pickle'
            ),
            'prep_resource': util.make_version_path(
                path=self.default_path['save'],
                module='prep',
                name='resource',
                version=self.fp_version,
                extension='pickle'
            ),
        }
