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

