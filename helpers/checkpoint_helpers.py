import pickle

def load_obj(name):
    with open('/home/john/objects/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
