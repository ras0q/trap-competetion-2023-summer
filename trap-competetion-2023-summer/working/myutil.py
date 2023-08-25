import re


def get_birth_year(birthday):
    if type(birthday) != str:
        return None
    pattern = r"\b\d{4}\b"
    matches = re.findall(pattern, birthday)
    if len(matches) > 1:
        raise ValueError("find twice yaer")
    elif len(matches) == 0:
        return None
    else:
        return int(matches[0])
