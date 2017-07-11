import numpy as np

class HMM:
    
    def __init__(self,y,N,observable_states,seed_pi=np.array([]),\
                 seed_a=np.array([]),seed_b=np.array([])):
        '''
        Attributes:

            y -- array of observations.
            
            N -- dimension of hidden states space.
            We don't know the spectrum of hidden states,
            instead we just enumerate them as 0...N-1
            
            observable_states -- array of size M of
            possible observations.
            
            M -- dimension of observed states space.
            
            pi -- probabilities of the initial states,
            initialized randomly on default.
            
            a -- transition matrix, hidden states
            (row) x hidden states (column), initialized
            randomly on default.
            
            b -- emission probabilities, hidden states
            (row) x observed states (column), initialized
            randomly on default.
            
            T -- length of time series.
            
            alpha -- forward probabilities,
            hidden states (row) x steps (column).
            
            beta -- backwards probabilities,
            hidden states (row) x steps (column).
            
            gamma -- hidden probabilities conditioned on
            observations, hidden states (row) x steps (column).
            
            xi -- double hidden probabilities conditioned on
            observtions, steps (base) x hidden states (row) x
            hidden states (column).
            
            y_indexes -- indexes of self.y in the
            self.observable_states. An equivalent way of storing
            observations 'y', used for all the matrix calculations.
            
            norms -- array of coefficients of
            normalization, of size T. Here we keep record
            of coefficients c_t=P(y_t|y_1...y_{t-1}), where
            c_0=P(y_0).
            
            R -- R-matrix of Viterbi algorithm, states x steps.
            
            Q -- Q-matrix of Viterbi algorithm, states x steps.
            
            estimate_x -- to store Viterbi estimate of the
            most likely hidden states.
        
        '''
        self.y=y
        
        self.N=N
        
        self.observable_states=observable_states
        
        self.M=len(self.observable_states)
        
        if not seed_pi.any():
            self.pi=np.random.random(size=self.N)
        else:
            self.pi=seed_pi
        
        if not seed_a.any():
            self.a=np.random.random(size=(self.N,self.N))
        else:
            self.a=seed_a
        
        if not seed_b.any():
            self.b=np.random.random(size=(self.N,self.M))
        else:
            self.b=seed_b
        
        self.T=len(self.y)
        
        self.alpha=np.zeros(shape=(self.N,self.T))
        
        self.beta=np.zeros(shape=(self.N,self.T))
        
        self.gamma=np.zeros(shape=(self.N,self.T))
        
        self.xi=np.zeros(shape=(self.T,self.N,self.N))
        
        self.y_indexes=self.indexes_of_observations()
        
        self.norms=[1 for _ in range(self.T)]
        
        self.R=np.zeros(shape=(self.N,self.T))
        
        self.Q=np.zeros(shape=(self.N,self.T))
        
        self.estimate_x=np.zeros(shape=self.T)
        '''
        Normalize all the initialized probabilities
        to sum up to one.
        '''
        self.normalize_a()
        self.normalize_b()
        self.normalize_pi()
        
    def normalize_a(self):
        '''
        Normalize initial hidden states transition
        probabilities in the 'a' matrix.
        '''
        for i in range(self.N):
            self.a[i]/=self.a[i].sum()
            self.a[i][-1]=1-self.a[i][:-1].sum()
        
    def normalize_b(self):
        '''
        Normalize initial emission
        probabilities in the 'b' matrix.
        '''
        for i in range(self.N):
            self.b[i]/=self.b[i].sum()
            self.b[i][-1]=1-self.b[i][:-1].sum()  
            
    def normalize_pi(self):
        '''
        Normalize initial probabailities of
        first step emissions.
        '''
        self.pi/=self.pi.sum()
        self.pi[-1]=1-self.pi[:-1].sum()
        
    def indexes_of_observations(self):
        '''
        We observe self.y which is a sequence of states from
        self.observable_states. This function returns sequence
        of indexes of self.y elements in self.observable_states.
        '''
        return [self.observable_states.index(obs) for obs in self.y]
        
    def forward(self):
        '''
        Calculate hat alpha using the forward algorithm.
        We calculate probability of hidden state "i" (row)
        at time step "t" (column) given the sequence of self.y
        up to the step "t", inclusive.
        '''        
        for i in range(self.N):
            self.alpha[i][0]=self.pi[i]*self.b[i][self.y_indexes[0]]        
        '''
        Normalize alpha[i][0] so that sum over "i" is 1,
        and save the corresponding normalization factor in
        the "norms" list.
        '''
        sum_alpha=0.0
        for i in range(self.N):
            sum_alpha+=self.alpha[i][0]
        self.norms[0]=sum_alpha 

        if sum_alpha==0:
            raise ValueError("All alpha at t=0 are zero")
        
        for i in range(self.N):
            self.alpha[i][0]/=sum_alpha

        '''
        Calculate alpha iteratively over "t".
        '''
        for t in range(self.T-1):
            for i in range(self.N):
                s=0
                for j in range(self.N):
                    s+=self.alpha[j][t]*self.a[j][i]
                s*=self.b[i][self.y_indexes[t+1]]
                self.alpha[i][t+1]=s            
            
            '''
            Normalize alpha[i][t+1] so that sum over "i" is 1,
            and save the corresponding normalization factor c_{t+1} in
            the "norms" list.
            '''
            sum_alpha=0.0
            for i in range(self.N):
                sum_alpha+=self.alpha[i][t+1]
                
            if sum_alpha==0:
                raise ValueError("All alpha at t={} are zero".format(t))                
                
            self.norms[t+1]=sum_alpha
            for i in range(self.N):
                self.alpha[i][t+1]/=sum_alpha
                
    def backward(self):
        '''
        Calculate hat beta using the backward algorithm.
        We calculate probability given hidden state "i" (row)
        at time step "t" (column) of the sequence of self.y
        after the step "t+1", inclusive. Betas calculated
        here are 'hat betas', renormalized using the c_t
        coefficients, determined while calculating
        alphas and saved in the "norms" list.
        '''           
        for i in range(self.N):
            self.beta[i][self.T-1]=1
        '''
        Calculate beta iteratively over "t".
        '''
        for t in range(self.T-1)[::-1]:
            for i in range(self.N):
                s=0
                for j in range(self.N):
                    s+=self.beta[j][t+1]*self.a[i][j]\
                    *self.b[j][self.y_indexes[t+1]]
                self.beta[i][t]=s            
            '''
            Normalize beta[i][t] by the same iverted factor as alpha
            '''
            for i in range(self.N):
                self.beta[i][t]/=self.norms[t+1]
            
    def BaumWelch(self):
        '''
        1. Calculate alpha and beta, using the forward/backward.
        2. Calculate gamma and xi, using alpha and beta.
        3. Calculate pi, a, and b, using gamma and xi.
        '''
        
        ## 1.        
        
        self.forward()
        self.backward()        
 
        ## 2. 
       
        for t in range(self.T):
            for i in range(self.N):
                self.gamma[i][t]=self.alpha[i][t]*self.beta[i][t]
                
        for t in range(self.T-1):
            for i in range(self.N):
                for j in range(self.N):
                    self.xi[t][i][j]=self.alpha[i][t]*self.beta[j][t+1]\
                    *self.a[i][j]*self.b[j][self.y_indexes[t+1]]
                    self.xi[t][i][j]/=self.norms[t+1]
        
        '''
        Check for the consistency of gamma and xi probabilities.
        We check whether sum_j xi_{tij}=gamma_{ti} for all t and i.
        '''
        for t in range(self.T-1):
            for i in range(self.N):
                s1=0.0
                for j in range(self.N):
                    s1+=self.xi[t][i][j]
                if abs(self.gamma[i][t]-s1)>0.0001:
                    raise ValueError("Gamma and Xi are inconsistent,\
                                     t={}".format(t))
                    
        '''
        Check for the consistency of gamma probabilities.
        We check whether sum_i gamma_{it}=1 for all t.
        '''
        for t in range(self.T-1):
            s1=0.0
            for i in range(self.N):
                s1+=self.gamma[i][t]
            if abs(s1-1)>0.0001:
                raise\
        ValueError("Gamma is inconsistent, t={}, sum={}".format(t,s1))

        ## 3. 
                    
        for i in range(self.N):
            self.pi[i]=self.gamma[i][0]
            
        for i in range(self.N):
            sum_gamma=0.0
            for t in range(self.T-1):
                sum_gamma+=self.gamma[i][t]
            for j in range(self.N):
                sum_xi=0.0
                for t in range(self.T-1):
                    sum_xi+=self.xi[t][i][j]
                self.a[i][j]=sum_xi/sum_gamma
                
        for i in range(self.N):
            for k in range(self.M):
                s_k=0.0
                s_all=0.0
                for t in range(self.T):
                    if self.y[t]==self.observable_states[k]:
                        s_k+=self.gamma[i][t]
                    s_all+=self.gamma[i][t]
                self.b[i][k]=s_k/s_all

    def Viterbi(self):
        '''
        Viterbi algorithm for the most likely sequence
        of hidden states. To be used when we know (or
        have estimated) the probability matrices a, b, pi.   
        The result is stored in self.estimate_x.
        
        1. Estimate R and Q matrices of probabilities
        used for Viterbi algorithm.
        2. Estimate the most likely hidden sequence.
        
        Return:
            void.
        '''
        for i in range(self.N):
            self.R[i][0]=self.pi[i]*self.b[i][self.y_indexes[0]]
            
        for t in range(1,self.T):
            for i in range(self.N):
                probs=[self.R[j][t-1]*self.a[j][i]\
                        for j in range(self.N)]
                self.R[i][t]=np.max(probs)*self.b[i][self.y_indexes[t]]
                self.Q[i][t]=probs.index(np.max(probs))
                
        probs=[self.R[i][self.T-1] for i in range(self.N)]                
        self.estimate_x[self.T-1]=probs.index(np.max(probs))
        for t in range(self.T-1)[::-1]:
            self.estimate_x[t]=self.Q[int(self.estimate_x[t+1])][t+1]