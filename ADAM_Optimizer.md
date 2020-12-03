# ADAM_Optimizer



with reference to https://arxiv.org/pdf/1412.6980v8.pdf

    
    
    
  
class ADAM_Optimizer:

    def __init__(self, weights, alpha=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.theta = weights
        self.m = 0           # initialize 1st moment estimate
        self.v = 0           # initialize 2nd moment estimate
        self.t = 0           # initialize time-step

    def backward_pass(self, gradient):
        self.t = self.t + 1
        self.m = self.beta1*self.m + (1 - self.beta1)*gradient                           # update 1st moment estimate
        self.v = self.beta2*self.v + (1 - self.beta2)*(gradient**2)                      # update 2nd moment estimate
        m_hat = self.m/(1 - self.beta1**self.t)                                          # create unbiased 1st moment estimate
        v_hat = self.v/(1 - self.beta2**self.t)                                          # create unbiased 2nd moment estimate
        self.theta = self.theta - self.alpha*(m_hat/(np.sqrt(v_hat) - self.epsilon))     # update objective parameters 
        return self.theta

