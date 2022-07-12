from common.name import Post
from common.option import Option

import datetime
import common.util as util
import datetime as dt


class Init(object):
    def __init__(self, io, query, default_path: dict):
        # io class instance attribute
        self.io = io
        self.query = query
        self.version = Option()
        self._post = Post()
        self._default_path = default_path
        self.pipeline_path = {}

        # Forward planning instance attribute
        self.fp_version = ''    # Factory planning version
        # self.fp_seq = fp_seq    # Factory planning sequence
        # self.fp_num = fp_num    # Factory planning number
        self.fp_seq = ''    # Factory planning sequence
        self.fp_num = ''    # Factory planning number
        self.start_day = ''

        # Time instance attribute
        self.calendar = None
        self.plant_start_day = None

    def run(self) -> None:
        self._get_version_info()
        self._set_calendar()
        self._set_fp_version()
        self._set_pipeline_path()
        self._set_plant_start_day()

    def _get_version_info(self) -> None:
        version = self.io.load_from_db(sql=self.query.sql_fp_eng_version()).squeeze()
        self.fp_version = version[self._post.fp_version]
        self.fp_seq = version[self._post.fp_seq]
        self.start_day = version['start_day']

    def _set_calendar(self) -> None:
        self.calendar = self.io.load_from_db(sql=self.query.sql_calendar())

    def _set_fp_version(self) -> None:
        self.version.set_version(fp_version=self.fp_version, fp_seq=self.fp_seq)

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
        self.plant_start_day = dt.datetime.strptime(self.start_day, '%Y%m%d')
