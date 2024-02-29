
from pprint import pprint


def test(**kwargs):
    for name in kwargs.keys():
        print(kwargs[name])

class Model():
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def dump(obj):
  for attr in dir(obj):
    print("obj.%s = %r" % (attr, getattr(obj, attr)))

if __name__ == '__main__':
    a = Model(a=1, b=2)
    print(a.a)
    
    
