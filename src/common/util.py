import os

def make_path(path: str, module: str, name: str, extension: str):
    path_dir = os.path.join(path, module)
    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, name + '.' + extension)

    return path


def make_version_path(path: str, module: str, version: str, name: str, extension: str):
    path_dir = os.path.join(path, module, version)

    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, version + '_' + name + '.' + extension)

    return path


def generate_name(name_list: list):
    return '@'.join(name_list)


def make_dir(path) -> None:
    if not os.path.isdir(path):
        os.mkdir(path)
