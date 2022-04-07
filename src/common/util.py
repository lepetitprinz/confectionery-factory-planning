import os
import string


def make_path(path: str, module: str, name: str, extension: str):
    path_dir = os.path.join(path, module)
    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, name + '.' + extension)

    return path


def make_version_path(path: str, module: str, name: str, version: str, extension: str):
    path_dir = os.path.join(path, module)

    # make the directory if directory does not exist
    if not os.path.isdir(path_dir):
        os.mkdir(path_dir)

    path = os.path.join(path_dir, name + '_' + version + '.' + extension)

    return path


def generate_name(name_list: list):
    return '@'.join(name_list)


def generate_route(from_nm: str, to_nm: str):
    route = from_nm + '-' + to_nm
    return route


def generate_alphabet_code(n, b):
    code = []
    alphabet = string.ascii_uppercase
    while True:
        code.append(alphabet[n % b])
        if n // b < b:
            code.append(alphabet[n // b])
            break
        n = n // b

    return ''.join(code[::-1])
