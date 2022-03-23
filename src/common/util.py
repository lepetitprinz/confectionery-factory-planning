import string


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
