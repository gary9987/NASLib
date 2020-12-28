class Predictor:
    
    def __init__(self, ss_type=None, encoding_type=None):
        self.ss_type = ss_type
        self.encoding_type = encoding_type
        
    def set_ss_type(self, ss_type):
        self.ss_type = ss_type
        
    def pre_process(self):
        """
        This is called at the start of the NAS algorithm,
        before any architectures have been queried
        """
        pass
    
    def fit(self, xtrain, ytrain):
        """
        This can be called any number of times during the NAS algorithm.
        input: list of architectures, list of architecture accuracies
        output: none
        """
        pass
    
    def query(self, xtest, info):
        """
        This can be called any number of times during the NAS algorithm.
        inputs: list of architectures, 
                info about the architectures (e.g., training data up to 20 epochs)
        output: predictions for the architectures
        """
        pass
    
    def requires_partial_training(self, xtest, fidelity):
        """
        Does the predictor require training the architecture partially?
        E.g., learning curve extrapolators.
        """
        pass