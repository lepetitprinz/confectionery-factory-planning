import os


def make_dir(path) -> None:
    if not os.path.isdir(path):
        os.mkdir(path)


def make_path(path: str, module: str, name: str, extension: str):
    path_dir = os.path.join(path, module)
    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, name + '.' + extension)

    return path


def make_vrsn_path(path: str, module: str, version: str, name: str, extension: str):
    path_dir = os.path.join(path, module, version)

    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, version + '_' + name + '.' + extension)

    return path


def make_fp_version_name(year, week, seq):
    return 'FP_' + year + week + '.' + seq


def generate_model_name(name_list: list):
    return '@'.join(name_list)


def assert_type_int(value):
    assert type(value) is int, 'Value is not int type'


def calc_daily_avail_time(day: int, day_time: int, night_time: int):
    if not isinstance(day_time, int):
        raise TypeError("Type of time is not integer")

    standard_time = day * 86400 + 43200
    start_time = standard_time - day_time
    end_time = standard_time + night_time

    return start_time, end_time


def save_log(log: list, path, version, name):
    # Set save directory
    save_dir = os.path.join(path, version)

    # make the directory if not exist
    make_dir(path=save_dir)

    # Sort log list
    log = sorted(log)

    # Save the log
    file = open(os.path.join(save_dir, 'log_' + name + '.txt'), 'w')
    for line in log:
        file.write(line + '\n')

    file.close()


def make_time_pair(data: list):
    pair = []
    temp = []
    for i, time in enumerate(data):
        if i % 2 == 0:
            temp = [time]
        else:
            temp.append(time)
            pair.append(temp)

    return pair

