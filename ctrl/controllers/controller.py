class StateFeedbackController:
    def compute_control(self, x, t):
        raise NotImplementedError

    def __call__(self, x):
        return self.compute_control(x, 0)
