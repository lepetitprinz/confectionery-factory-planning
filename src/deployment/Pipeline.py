import common.util as util
from dao.DataIO import DataIO
from common.SqlConfigEngine import SqlConfig
# from common.SqlConfig import SqlConfig
from init.Init import Init
from init.DataLoad import DataLoad
from init.Preprocess import Preprocess
from model.OptSeqModel import OptSeqModel
from model.PostProcess import PostProcess


class Pipeline(object):
    def __init__(self, step_cfg: dict, exec_cfg: dict, default_path: dict):
        self.io = DataIO()
        self.sql_conf = SqlConfig()
        self.step_cfg = step_cfg    # Step configuration
        self.exec_cfg = exec_cfg    # Execution configuration

        # Path instance attribute
        self.default_path = default_path
        self.pipeline_path = {}

        # Factory information instance attribute
        self.fp_version = ''
        self.plant_start_time = None

    def run(self):
        # ============================================= #
        # 1. Initialization dataset
        # ============================================= #
        # Instantiate init class
        print("Step: Initialize engine information")
        init = Init(io=self.io, sql_conf=self.sql_conf, default_path=self.default_path)
        init.run()

        # Set initialized object
        self.pipeline_path = init.pipeline_path
        self.fp_version = init.fp_version
        self.plant_start_time = init.plant_start_time

        # ============================================= #
        # 2. Load dataset
        # ============================================= #
        # Instantiate load class
        load = DataLoad(io=self.io, sql_conf=self.sql_conf)

        master, demand = None, None
        if self.step_cfg['cls_load']:
            master = load.load_master()    # Master dataset
            demand = load.load_demand()    # Demand dataset

            # Save the master & demand information
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=master, file_path=self.pipeline_path['load_master'], data_type='binary')
                self.io.save_object(data=demand, file_path=self.pipeline_path['load_demand'], data_type='binary')

        # ============================================= #
        # 3. Data preprocessing
        # ============================================= #
        dmd_prep, res_prep = (None, None)
        if self.step_cfg['cls_prep']:
            print("Step: Data Preprocessing")

            if not self.step_cfg['cls_load']:
                master = self.io.load_object(file_path=self.pipeline_path['load_master'], data_type='binary')
                demand = self.io.load_object(file_path=self.pipeline_path['load_demand'], data_type='binary')

            # Instantiate data preprocessing class
            prep = Preprocess()

            # Preprocess demand & resource dataset
            dmd_prep = prep.set_dmd_info(data=demand)    # Demand
            res_prep = prep.set_res_info(data=master)    # Resource

            # Save the preprocessed demand
            if self.exec_cfg['save_step_yn']:
                self.io.save_object(data=dmd_prep, file_path=self.pipeline_path['prep_demand'], data_type='binary')
                self.io.save_object(data=res_prep, file_path=self.pipeline_path['prep_resource'], data_type='binary')

        # ============================================= #
        # Model
        # ============================================= #
        if self.step_cfg['cls_model']:
            print("Step: Modeling")

            if not self.step_cfg['cls_prep']:
                dmd_prep = self.io.load_object(file_path=self.pipeline_path['prep_demand'], data_type='binary')
                res_prep = self.io.load_object(file_path=self.pipeline_path['prep_resource'], data_type='binary')

            # Model by plant
            for plant in dmd_prep['plant_dmd_list']:
                opt_seq = OptSeqModel(
                    plant=plant,
                    dmd_due=dmd_prep['plant_dmd_due'][plant],
                    res_info=res_prep
                )

                # Instantiate model
                model = opt_seq.init(
                    dmd_list=dmd_prep['plant_dmd_list'][plant],
                    res_grp_list=res_prep['plant_res_grp'][plant]
                )

                if self.exec_cfg['save_step_yn']:
                    self.io.save_object(
                        data=model,
                        file_path=util.make_version_path(
                            path=self.default_path['save'],
                            module='model',
                            name='model_' + plant,
                            version=self.fp_version,
                            extension='pickle'
                        ),
                        data_type='binary'
                    )

                # opt_seq.optimize(model=model)

        # ============================================= #
        # Post Process
        # ============================================= #
        if self.step_cfg['cls_pp']:
            # Post Process after optimization
            pp = PostProcess()
            pp.post_process()
