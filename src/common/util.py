import string


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
