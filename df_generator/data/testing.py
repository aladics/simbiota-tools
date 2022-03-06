import itertools

def dict_product(**dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


a = {}
a["param1"] = [1,2,3]
a["param2"] = [3,4,5]

print(list(dict_product(**a)))