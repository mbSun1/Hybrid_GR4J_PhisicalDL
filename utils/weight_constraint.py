class weightConstraint(object):
    def __init__(self):
        pass

    def __call__(self, module):
        if hasattr(module, 'hydropara'):
            w = module.hydropara.data
            w = w.clamp(0, 1)
            module.hydropara.data = w