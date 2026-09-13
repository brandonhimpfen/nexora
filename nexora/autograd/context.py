class Context:
    def __init__(self):
        self.saved_tensors = ()
        self.meta = {}

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
