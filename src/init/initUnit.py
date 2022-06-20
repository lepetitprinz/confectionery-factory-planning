from common.option import Option

import datetime
import common.util as util
import datetime as dt


class Init(object):
    def __init__(self, io, query, default_path: dict, fp_num: str, fp_seq: str):
        # io class instance attribute
        self.io = io
        self.query = query
        self.version = Option()
        self._default_path = default_path
        self.pipeline_path = {}

        # Forward planning instance attribute
        self.fp_version = ''    # Factory planning version
        self.fp_seq = fp_seq    # Factory planning sequence
        self.fp_num = fp_num    # Factory planning number

        # Time instance attribute
        self.calendar = None
        self.plant_start_day = None

    def run(self) -> None:
        self._set_calendar()
        self._set_fp_version()
        self._set_pipeline_path()
        self._set_plant_start_day()

    def _set_calendar(self) -> None:
        self.calendar = self.io.load_from_db(sql=self.query.sql_calendar())

    def _set_fp_version(self) -> None:
        # Get days
        today = dt.date.today().strftime('%Y%m%d')

        # Set forward planning version
        today_df = self.calendar[self.calendar['yymmdd'] == today].copy()
        # fp_version = util.make_fp_version_name(
        #     year=today_df['yy'].values[0],
        #     week=today_df['week'].values[0],
        #     seq=self.fp_num
        # )
        fp_version = 'unit_test'    # Todo: temporal

        self.fp_version = fp_version
        self.version.set_version(fp_version=fp_version, fp_seq=self.fp_seq)

    # Set the pipeline path
    def _set_pipeline_path(self) -> None:
        self.pipeline_path = {
            'load_data': util.make_vrsn_path(
                path=self._default_path['save'],
                module='load',
                version=self.fp_version,
                name='data' + self.fp_seq,
                extension='pickle'
            ),
            'load_master': util.make_vrsn_path(
                path=self._default_path['save'],
                module='load',
                version=self.fp_version,
                name='master_' + self.fp_seq,
                extension='pickle'
            ),
            'load_demand': util.make_vrsn_path(
                path=self._default_path['save'],
                module='load',
                version=self.fp_version,
                name='demand_' + self.fp_seq,
                extension='pickle'
            ),
            'prep_data': util.make_vrsn_path(
                path=self._default_path['save'],
                module='prep',
                version=self.fp_version,
                name='data_' + self.fp_seq,
                extension='pickle'
            ),
            'model': util.make_vrsn_path(
                path=self._default_path['save'],
                module='model',
                version=self.fp_version,
                name='model_' + self.fp_seq,
                extension='pickle'
            ),
        }

    # Set plant start day
    def _set_plant_start_day(self) -> None:
        today = dt.datetime.combine(dt.datetime.today(), dt.datetime.min.time())
        today = today - datetime.timedelta(days=today.weekday())    # This monday

        self.plant_start_day = today
