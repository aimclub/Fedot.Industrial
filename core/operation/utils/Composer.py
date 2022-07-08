class FeatureGeneratorComposer:
    def __init__(self):
        self.dict = {}

    def __getitem__(self, item):
        return self.dict[item]

    def add_operation(self, operation_name: str,
                      operation_functionality: object):
        self.dict[operation_name] = operation_functionality
