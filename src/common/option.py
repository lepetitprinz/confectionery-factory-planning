import common.config as config


class Option(object):
    def __init__(self):
        # Version
        self.fp_seq = ''
        self.fp_version = ''

        # Time
        self.work_day = config.work_day
        self.time_uom = config.time_uom

        # Model parameter
        self.time_limit = config.time_limit
        self.make_span = config.make_span
        self.optput_flag = config.optput_flag
        self.max_iteration = config.max_iteration

    def set_version(self, fp_version, fp_seq):
        self.fp_seq = fp_seq
        self.fp_version = fp_version
