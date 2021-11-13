import numpy as np
import math
from scipy.stats import chisquare
from scipy.stats import poisson
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture # For Expectation maximisation algorithm


#dependencies: numpy, scipy (for scipy.stats.chisquare), sklearn (for sklearn.preprocessing.MinMaxScaler)
class P3C:
 # General class functionality (Robert)
    
    #class variables
    _alpha = 0.0001 #alpha for chi-squared-test
    _EM_iter = 10 #iterations for EM

    #Poisson threshold is the only parameter for P3C 
    def __init__(self, poisson_threshold): 
        
        #sklearn-like attributes
        self.labels_ = [] 
        self.cluster_centers_ = None #"cluster cores"
        
        #internally used variables
        self._X = None
        self._poisson_threshold = poisson_threshold
        self._support_set = []
        self._supports = []
        self._approx_proj = []
        self._support_set_of_cores = []
        self._fuzzy_membership_matrix = None
        self._hard_membership_matrix = None #np.array: dimension: number of samples x number of clusters
        
    
    def __convert_matrix_to_labels (self):
        """Converts membership matrix to labels"""
        
        for sample in self._membership_matrix:
            for entry in range(sample.size):
                if sample[entry] == 1:
                    self.labels_.append(entry)
    
    # Methods for 3.1: Projections of true p-signatures (Mahdi and Robert)

    def __normalize(self, X):
        '''Normalizes the data featurewise

        Parameters
        ----------
        self, 

        X : numpy.array of the input data 

        Returns
        -------
        numpy.array of the normalized data
        
        '''
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def __uniformity_test(self, attr):
        '''Uses the chi-squared test to determine if the attribute is uniformly disributed,
        using the class variable self._alpha as a certrainty-threshold.

        Parameters
        ----------
        self, 

        attr: input list to be tested for uniformity 

        Returns
        -------
        True, if input data is uniform

        '''
        
        if chisquare(attr)[0] > self._alpha:
            return False
        return True
        
    
    def __compute_support(self):
        '''Computes Supports of each bin
        This function computes support set of each interval S and its support.
        then assigns the values to self._support_set = [] and self._supports = []
        SupportSet(S)= {x ∈ D | x.aj ∈ S }
        Support(S) = |SupportSet(S)|


        Parameters
        ----------
        self, 

        M : numpy.array 

        Returns
        -------

        '''
        n = self._X.shape[0] # n = number of data objects
        attribute_number = self._X.shape[1] # number of attributes 
        bin_number = int(1 + math.log(n,2)) # number of bins


        for i in range(attribute_number):
            # support set of each interval S for attribiute i
            supp_set = [[]for i in range(bin_number)]

            interval_length = 1 / bin_number

            # calculate in which bin should the point be placed based on attribute i
            for j in range(n):
                supp_set_index = math.floor(self._X[j,i]/interval_length)

                if supp_set_index == len(supp_set):
                    supp_set_index-=1 

                supp_set[supp_set_index].append(self._X[j,:])

            self._support_set.append(supp_set)
            

        self._supports = [[] for i in range(len(self._support_set))]
        for i in range(len(self._support_set)):
            for j in range(len(self._support_set[i])):
                self._supports[i].append(len(self._support_set[i][j]))

        return 
    
    def __approximate_projection(self):
        
        '''Finds the bins that violate uniform distribution from the data stored in self._supports and
        storest them in the numpy.array bins, with 1 for non-uniformity and 0 for uniformity. It then 
        assignes the intervals to self._approx_proj

        Parameters
        ----------
        self

        Returns
        -------

        '''
        
        bins = np.zeros((len(self._supports), len(self._supports[0])), dtype=int)
        for attr_number in range(len(self._supports)): #loop over all attributes
            #print ("attr_nr =", attr_number)
                            
            supp = self._supports[attr_number].copy() #make a copy of the supports of the current attribute
            while self.__uniformity_test(supp) == False: #if not uniform find highest element
                max_index = supp.index(max(supp))
                supp.pop(max_index) #remove highest element from list
                #print (supp)
                
                i = 0 
                while i <= max_index: #loop to adjust max_index according to previousely deleted elements
                    max_index += bins[attr_number, i]
                    i += 1
                bins[attr_number, max_index]  = 1 #mark highest bin
                #print ("current bin: ", bins[attr_number])
            
            interval_list = [] #2d list for current attribute
            interval = [] #current interval
            open_interval = False

            for i in range(len(bins[attr_number])):
                if open_interval == False: #open new interval
                    if bins[attr_number, i] == 1:
                        interval.append(i/len(bins[attr_number]))
                        open_interval = True
                if open_interval == True: #close current interval
                    if bins[attr_number, i] == 0:
                        interval.append(i/len(bins[attr_number]))
                        interval_list.append (interval)
                        interval = []
                        open_interval = False
                    if (i == len(bins[attr_number])-1) and (bins[attr_number, i] == 1): #last bin marked 1
                        interval.append ((i+1)/len(bins[attr_number]))
                        interval_list.append (interval)                

            self._approx_proj.append (interval_list)
            
    
    # Methods for 3.2: Cluster Cores (Akshey and Jonas)
    
    def __convert_approx_proj_to_dict(self):
        ''' Converts _approx_proj from 3.2 to a dictonary
        This is necessary to compute cluster cores with apriori

        Parameters
        ----------
        self._approx_proj : three nested lists (# attribut/# interval/ start and end of interval)

        Returns
        -------
        _approx_proj_sig : list of dictonaries, each dictonary is a projection e.g {0: [0.1, 0.2]}

        '''
        _approx_proj_sig = []
        for attribute, row in enumerate(self._approx_proj):
            for interval in row:
                _approx_proj_sig.append({attribute:interval})
                
        return _approx_proj_sig
    
    
    def __compute_support_sig(self, p_signature):
        '''Computes support for p-signature
        This function computes the support by removing data points
        that do not lie in any of the intervals of the given p-signature

        Parameters
        ----------
        p_signature : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        dataset : numpy.ndarray 

        Returns
        -------
        data.shape[0] : number of points in p-signature

        '''

        data = np.copy(self._X)
        for attribute in p_signature:
            interval = p_signature[attribute]
            remove = []
            for i, point in enumerate(data):
                if  interval[0] > point[attribute] or point[attribute] > interval[1]:
                    remove.append(i)
            data = np.delete(data, remove, 0)

        return data.shape[0]

    
    def __compute_exp_support(self, p_signature, interval):
        ''' Computes expected support for a p-signature

        Parameters
        ----------
        p-signature : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        interval : list with start and end value of interval

        data : normalized np.ndarray

        Returns
        -------
        support * width : 

        '''

        support = self.__compute_support_sig(p_signature)
        width = abs(interval[0] - interval[1])

        return support*width
    
    
    def __diff_interval(self, p_signature, pplus1_signature):
        '''Helper function to compute difference in interval for two p-signatures.
           Used for possion threshold

        Parameters
        ---------- 
        p_signature : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        pplus1_signature : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        Returns
        -------
        interval : 

        '''

        diff = list(set(pplus1_signature) - set(p_signature))
        interval = pplus1_signature[diff[0]]
        
        return interval



    def __check_core_condition(self, p_signature, pplus1_signature):
        ''' Checks if probability is smaller than possion threshold: 
        Possion(Supp(k+1 signature), ESupp(k+1 signature)) < possion_threshold
        Returns True is poisson value is smaller than threshold

        and

        Checks if support is larger than expected support: 
        Supp(k+1 signature) > ESupp(k+1 siganature)
        ESupp = Supp(S) * width(S')

        Parameters
        ----------
        p-signatue : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        pplus1_signature : dictronary e.g. {0:[0,0.1], 3:[0.1,0.2]} -> Intervals for attributes 0 and 3

        dataset : numpy.ndarray

        threshold : poisson_threshold -> defined by user. default: 1e-20

        Returns
        -------
        true/false : 

        '''
        
        interval = self.__diff_interval(p_signature, pplus1_signature)
        support = self.__compute_support_sig(pplus1_signature)
        expected_support = self.__compute_exp_support(pplus1_signature, interval)
        base_condition = support > expected_support
        if base_condition:
            poisson_value = poisson.pmf(support, expected_support) 
            if poisson_value < self._poisson_threshold:
                return True
            else:
                return False
            
            
    def __merge(self, dict1, dict2):
        '''Helper function to merge to p-signature dictonaries 

        Parameters
        ----------
        dict1: p-signature dictonary

        dict2: p-signature dictronary 

        Returns
        -------
        res : merged p-signature containing of dict1 and dict 2

        '''
        res = {**dict1, **dict2}
        return res

    
    def __a_is_subset_of_b(self, a,b):
        '''Helper function that checks is dictionary a is a subset of dictionary b 

        Parameters
        ----------
        a : dictionary

        b : dictionary

        Returns
        -------
        bool : truth value

        '''
        return all((k in b and b[k]==v) for k,v in a.items())
    
    
    def __apriori_cores(self, _approx_proj_sig):
        ''' Computes cluster cores in apriori fashion. 
        The function computes maximal p-signatures that fulfill 
        two conditions. 

        Parameters
        ----------
        _approx_proj_sig : list of dictonaries

        data: dataset np.ndarray


        Returns
        -------
        cluster_centers_ : list of p-signatures e.g. [{0: [0.1, 0.2]}, {1: [0.5, 0.8], 2: [0.1, 0.4], 3: [0.5, 0.7]}]

        '''

        # Loop through attributes and intervals (ignore same dimensions)
        _cluster_cores = [_approx_proj_sig]

        while _cluster_cores[-1] != []:
            p_sig_list = []
            for p_sig in _cluster_cores[-1]:
                for one_sig in _approx_proj_sig:
                  ### The second criterion is to avoid double counting
                  ### In case this second criteria gives errors we could make a set of the p-signatures before checking the core condition
                    if list(one_sig.keys())[0] not in p_sig.keys() and list(one_sig.keys())[0] > max(list(p_sig.keys())): 
                        pplus1_sig = self.__merge(p_sig, one_sig)
                        ### Check core condition
                        if self.__check_core_condition(p_sig, pplus1_sig):
                            p_sig_list.append(pplus1_sig)
            _cluster_cores.append(p_sig_list)
        _cluster_cores.pop()

        # Finds only the unique cluster cores since the above algorithm might be return the same core multiple times 
        cluster_centers_ = []
        for p_sig_list in _cluster_cores:
            for p_sig in p_sig_list:
                if p_sig not in cluster_centers_:
                    cluster_centers_.append(p_sig)

        maximal_cluster_centers_ = cluster_centers_.copy()
        # Check condition 2 for each signature (pruning to maximal cluster cores)
        for cluster in reversed(cluster_centers_):
            for sub_cluster in reversed(cluster_centers_):
                if cluster != sub_cluster: 
                    if self.__a_is_subset_of_b(sub_cluster, cluster):
                        if sub_cluster in maximal_cluster_centers_:
                            maximal_cluster_centers_.remove(sub_cluster)


        self.cluster_centers_ = maximal_cluster_centers_

    """
    def __compute_core_support(self, cores_p_signatures):
        ''' Computes the support number for each cluster core.

        Parameters
        ----------
        cores_p_signatures : list of core signatures e.g. [{0: [0.1, 0.2]}, {1: [0.5, 0.8], 2: [0.1, 0.4], 3: [0.5, 0.7]}]

        Returns
        -------
        cores_support_number : list of support numbers for each cluster core

        '''
        cores_support_number = []
        for p_sig in cores_p_signatures:
            cores_support_number.append(__compute_support_sig(p_sig))
        
        return cores_support_number
    """
    
    def __compute_core_set(self):
        ''' Computes the support set for each cluster core. This is necessary for 3.3:
        computing the projected clusters 

        Parameters
        ----------
        cores_p_signatures : list of core signatures e.g. [{0: [0.1, 0.2]}, {1: [0.5, 0.8], 2: [0.1, 0.4], 3: [0.5, 0.7]}]

        Returns
        -------
        cores_set : list of support sets for each cluster core

        '''
        
        for p_signature in self.cluster_centers_:
            dataset = np.copy(self._X)
            for attribute in p_signature:
                interval = p_signature[attribute]
                remove = []
                for i, point in enumerate(dataset):
                    if  interval[0] > point[attribute] or point[attribute] > interval[1]:
                        remove.append(i)
                dataset = np.delete(dataset, remove, 0)
            self._support_set_of_cores.append(dataset)
    
     # Methods for 3.3: Computing projected clusters (Manju & Mahdi)
    def __compute_fuzzy_membership_matrix(self):
        '''
        Refines the cluster cores into projected clusters
        Parameters
        ----------
        cluster_core_i:
        
        data:
        
        Returns
        -------
        fuzzy_membership_matrix: 
        
        '''
        
        n = self._X.shape[0]
        k = len(self.cluster_centers_)
        
        self._fuzzy_membership_matrix = np.zeros((n, k))
        for i in range(0,n):
            for l in range(0,k):
                if (any(np.array_equal(self._X[i], x) for x in  self._support_set_of_cores[l])):
#                     mil is equal to the fraction of clusters cores that 
#                     contain data point i in their support set, 
#                     if i is in the support set of cluster core l.
#                     fuzzy[i][l] = cluster cores with i / cluster cores
                    counter = 0
    
                    
                    for j in range(len(self._support_set_of_cores)):
                        if (any(np.array_equal(self._X[i], x) for x in  self._support_set_of_cores[j])):
                            counter += 1
                        
                    self._fuzzy_membership_matrix[i][l] = 1
        
        
                    
            """       
        #pre_matrix = np.random.randint(2, size=(100,5)) ###100 objects, 5 cluster cores
        fraction_matrix = np.sum(self._fuzzy_membership_matrix, axis=1)/k
        fraction_matrix = fraction_matrix.reshape(n,1)
        self._fuzzy_membership_matrix = np.multiply(self._fuzzy_membership_matrix, fraction_matrix) 
        """
    
    def __compute_hard_membership_matrix(self):
        '''
          For each data point compute the probability of belonging to each projected cluster using Expectation 
          Maximization(EM)algorithm.
          Parameters
          ----------
          fuzzy_membership_matrix:
          
          max_iterations:
          
          Returns
          -------
          membership_matrix:
        
        '''
        
        gm = GaussianMixture(n_components=self._fuzzy_membership_matrix.shape[1]).fit(self._fuzzy_membership_matrix)
        self.labels_ = gm.predict(self._fuzzy_membership_matrix)

        
        


    #data X is numpy.ndarray with samples x features (no label!)  
    def fit (self, X):
        self._X = self.__normalize(X)
        self.__compute_support()
        self.__approximate_projection()
        self.__apriori_cores(self.__convert_approx_proj_to_dict())
        self.__compute_core_set()
        
        
        #self.__convert_matrix_to_labels()
        
        
        
        #self.cluster_center_ = ... #used as interface between part 3.2. and 3.3
        
    #data X is numpy.ndarray with samples x features (no label!)  
    def predict (self, X):
        self.__compute_fuzzy_membership_matrix()
        self.__compute_hard_membership_matrix()
        
        #all the method calls of 3.3, 3.4 and 3.5 we have to implement go here...
        
        #self.labels_ = #final result of the algorithm.
        
    #data X is numpy.ndarray with samples x features (no label!)  
    def fit_predict (self, X):
        self.fit (X)
        self.predict (X)