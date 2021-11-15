import numpy as np
import math
import warnings
from scipy import stats
from scipy.stats import chisquare
from scipy.stats import poisson
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture 
from sklearn.metrics import normalized_mutual_info_score
from sklearn.exceptions import ConvergenceWarning

class P3C:
 # General class functionality (Robert)
    
    #class variables
    _alpha = 0.001 #alpha for chi-squared-test
    _EM_iter = 10 #iterations for EM

    #Poisson threshold is the only parameter for P3C 
    def __init__(self, poisson_threshold): 
        
        #sklearn-like attributes
        self.labels_ = [] 
        self.cluster_centers_ = None #"cluster cores"
        
        #internally used variables
        self._X = None
        self._poisson_threshold = poisson_threshold
        self._support_set = [] #for the bins
        self._supports = [] #for the bins
        self._approx_proj = [] #approximate projections
        self._support_set_of_cores = []
        self._fuzzy_membership_matrix = None
    
    # Methods for 3.1: Projections of true p-signatures (Mahdi and Robert)

    def __normalize(self, X):
        '''Normalizes the data featurewise
        Parameters
        ----------
        X : numpy.array of the input data  
        
        Returns
        -------
        Normalized data
        '''
        
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def __uniformity_test(self, attr):
        '''Uses the chi-squared test to determine if the attribute is uniformly disributed,
        using the class variable self._alpha as a certrainty-threshold.

        Parameters
        ----------

        attr: input list to be tested for uniformity 

        Returns
        -------
        True, if input data is uniform
        '''
        
        if not np.any(attr):
            return True
        if chisquare(attr)[0] > self._alpha:
            return False
        return True
    
    def __compute_support(self):
        '''Computes Supports of each bin'''
        
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
    
    def __approximate_projection(self):
        
        '''Finds the bins that violate uniform distribution from the data stored in self._supports and
        storest them in the numpy.array bins, with 1 for non-uniformity and 0 for uniformity. It then 
        assignes the intervals to self._approx_proj'''
        
        bins = np.zeros((len(self._supports), len(self._supports[0])), dtype=int)
        for attr_number in range(len(self._supports)): #loop over all attributes
                            
            supp = self._supports[attr_number].copy() #make a copy of the supports of the current attribute
            while self.__uniformity_test(supp) == False: #if not uniform find highest element
                max_index = supp.index(max(supp))
                supp.pop(max_index) #remove highest element from list
                
                i = 0 
                while i <= max_index: #loop to adjust max_index according to previousely deleted elements
                    max_index += bins[attr_number, i]
                    i += 1
                bins[attr_number, max_index]  = 1 #mark highest bin
            
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
        ''' Converts _approx_proj to a dictonary
        This is necessary to compute cluster cores with apriori'''
        
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

        Returns
        -------
        support * width
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
        interval
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

        Returns
        -------
        True, if core condition is met 
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

    def __a_is_subset_of_b(self, a, b):
        '''Helper function that checks is dictionary a is a subset of dictionary b 

        Parameters
        ----------
        a : dictionary

        b : dictionary

        Returns
        -------
        True, if a is a subset of b
        '''
        
        return all((k in b and b[k]==v) for k,v in a.items())
    
    def __apriori_cores(self, _approx_proj_sig):
        ''' Computes cluster cores in apriori fashion. 
        The function computes maximal p-signatures that fulfill 
        two conditions. 

        Parameters
        ----------
        _approx_proj_sig : list of dictonaries
        '''

        # Loop through attributes and intervals (ignore same dimensions)
        _cluster_cores = [_approx_proj_sig]

        while _cluster_cores[-1] != []:
            p_sig_list = []
            for p_sig in _cluster_cores[-1]:
                for one_sig in _approx_proj_sig:
                  # The second criterion is to avoid double counting
                    if list(one_sig.keys())[0] not in p_sig.keys() and list(one_sig.keys())[0] > max(list(p_sig.keys())): 
                        pplus1_sig = self.__merge(p_sig, one_sig)
                        # Check core condition
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
    
    def __compute_core_set(self):
        ''' Computes the support set for each cluster core. This is necessary for 3.3:
        computing the projected clusters '''
        
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
    
    
     # Methods for 3.3: Computing projected clusters (Mahdi and Robert)
    def __compute_fuzzy_membership_matrix(self):
        '''Computes the fuzzy membership matrix using the support set of cores and cluster centers.
        Then it assignes unassigned points to the closest cluster core using Mahalanobis distance'''
        
        n = self._X.shape[0] #number of data points
        k = len(self.cluster_centers_) #number of clusters
        
        self._fuzzy_membership_matrix = np.zeros((n, k)) #start fuzzy membership with an n*k array of zeros
        
        for i in range(n):
            for l in range(k):
                #for each point that belong to support set of a cluster assign 1 to fuzzy matrix
                if (any(np.array_equal(self._X[i], x) for x in  self._support_set_of_cores[l])):
                    
                    self._fuzzy_membership_matrix[i][l] = 1
        #compute fraction of data points that have that point in their support set           
        fraction_matrix = np.sum(self._fuzzy_membership_matrix, axis=1)/k
        fraction_matrix = fraction_matrix.reshape(n,1)
        self._fuzzy_membership_matrix = np.multiply(self._fuzzy_membership_matrix, fraction_matrix) 
        
        #unassigned data points are assigned to the “closest” cluster core
        #in terms of Mahalanobis distances to means of support sets of cluster cores.
        for i in range(n):
            m_distance = []
            #find rows of zero in fuzzy matrix
            if(not np.any(self._fuzzy_membership_matrix[i])):
                for l in range (0,len(self._support_set_of_cores)):
                    #mean of support sets
                    mean = sum(self._support_set_of_cores[l])/len(self._support_set_of_cores[l])
                    #inverse covariance
                    V = np.cov(np.array([self._X[i], mean]).T)
                    IV = np.linalg.inv(V)
                    #compute distance
                    m_distance.append(distance.mahalanobis(self._X[i], mean, IV))
                #assign to the smallest distance
                self._fuzzy_membership_matrix[i][m_distance.index(min(m_distance))] = 1
        
    def __compute_hard_membership_matrix(self):
        '''
          For each data point compute the probability of belonging to each projected cluster using Expectation 
          Maximization(EM)algorithm.'''

        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        gm = GaussianMixture(n_components=self._fuzzy_membership_matrix.shape[1]).fit(self._fuzzy_membership_matrix)
        self.labels_ = gm.predict(self._fuzzy_membership_matrix)
        
        
    # Methods for 3.4: Outlier detection (Akshey and Jonas)
    def __outlier_detection(self):
        '''Computes oultiers depending on mahalanobis distance
        Sets the labels for the computed outliers to -1'''
        
        cluster_outlier = []
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        # Loop through cluster cores
        for i in range(max(self.labels_)):
            outlier_list = []
            # Compute Covarianve matrix for each cluster
            cluster_cov = np.cov(self._X[self.labels_==i].T)
            # Compute inverse of covariance matrix
            if np.linalg.cond(cluster_cov > 1000):
                return
            cluster_cov_inv = np.linalg.inv(cluster_cov)
            # Computer mean of cluster core
            cluster_mean = np.mean(self._X[self.labels_==i] , axis=0)
            distances = []
            # Loop through each data point in cluster core and compute mahalanobis
            for point in self._X[self.labels_==i]:
                maha_distance = distance.mahalanobis(point, 
                                cluster_mean, cluster_cov_inv)
                distances.append(maha_distance)

            # Compute distance threshold for outlier detection, OR IS IT JUST alpha??? 
            #threshold = chisquare (self._X[self.labels_==i].T.shape[0])
            threshold = stats.chi2.ppf(1-self._alpha, self._X[self.labels_==i].T.shape[0])
            #threshold = stats.chi2.ppf(self._alpha, self._X[self.labels_==i].T.shape[0])
            outlier_indx = np.where(distances > threshold)[0].tolist()
            # Get original data index for outlier
            cluster_data = self._X[self.labels_==i]
            outlier_points = cluster_data[outlier_indx]
            for outlier_point in outlier_points:
                indx = np.where(self._X==outlier_point)[0][0].tolist()
                outlier_list.append(indx)
            cluster_outlier.append(outlier_list)
        flat_list = [item for sublist in cluster_outlier for item in sublist]
        for i in flat_list:
            self.labels_[i] = -1
        
    #sklearn-like functions       

    def fit (self, X):
        '''Fits the model according to X 

        Parameters
        ----------
        X : dataset
        '''
        self._X = self.__normalize(X)
        self.__compute_support()
        self.__approximate_projection()
        self.__apriori_cores(self.__convert_approx_proj_to_dict())
        self.__compute_core_set()
        
    def predict (self, X):
        '''Predicts labels of X according to the model and writes them to labels_, where they can be accessed

        Parameters
        ----------
        X : dataset
        '''
        self._X = self.__normalize(X)
        self.__compute_fuzzy_membership_matrix()
        self.__compute_hard_membership_matrix()
        self.__outlier_detection()
        
    def fit_predict (self, X):
        self.fit (X)
        self.predict (X)