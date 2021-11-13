# Script for running an elki algorithm and save its results
import os
import re
from subprocess import Popen, PIPE
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import normalized_mutual_info_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

DATA_FILE_NAME = "data_elki.tsv"
# Install Java and download the elki bundle https://elki-project.github.io/releases/release0.7.5/elki-bundle-0.7.5.jar
ELKI_JAR = "elki-bundle-0.7.5.jar"

def run_elki_clustering(arg_list, X):
    """Perform a clustering implemented by ELKI package.
       The function calls jar package, which must be accessible through the
       path stated in ELKI_JAR constant.

        Parameters
        ----------
        arg_list: list of strings with elki parameters
        X : array of shape (n_samples, n_features)
            A feature array.

        Returns
        -------
        labels : array [n_samples]
            Cluster labels for each point.
    """
    # write data into tsv file
    np.savetxt(DATA_FILE_NAME, X, delimiter=",", fmt="%.6f")
    # run elki with java
    process = Popen(arg_list,
                    stdout=PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        raise IOError("Elki implementation failed to execute: \n {}".format(output.decode("utf-8")))

    # remove data file
    os.remove(DATA_FILE_NAME)

    # parse output
    elki_output = output.decode("utf-8")
    # print(elki_output)
    # initialize array of ids and labels
    Y_pred = np.array([]).reshape(0, 2)
    
    # for each cluster, split by regex from output
    cluster_type = "Cluster: "
    splitted = elki_output.split(cluster_type)[1:]    
    for i, cluster in enumerate(splitted):
        # find point coordinates in output^
        # print(f"i: {i}\n\n cluster: {cluster}")
        IDs_list = re.findall(r"ID=(\d+)", cluster)
        # create a numpy array
        IDs = np.array(IDs_list, dtype="i").reshape(-1, 1)
        # append label
        IDs_and_labels = np.hstack((IDs, np.repeat(i, len(IDs_list)).reshape(-1, 1)))
        # append to matrix
        Y_pred = np.vstack((Y_pred, IDs_and_labels))

    # sort by ID, so that the points correspond to the original X matrix
    Y_pred = Y_pred[Y_pred[:, 0].argsort()]
    # remove ID
    return Y_pred[:, 1].astype(int)

def run_elki_P3C(X, poisson):
    # You can read of the names of the parameters for the algorithm from the elki GUI or the java code
    arg_list = ["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.subspace.P3C",
                     "-dbc.in", DATA_FILE_NAME,
                     "-parser.colsep", ",",
                     "-p3c.threshold", str(poisson)
                    ]
    predicted_labels = run_elki_clustering(arg_list, X)
    return predicted_labels


def run_elki_dbscan(X, epsilon, minpts):
    # You can read of the names of the parameters for the algorithm from the elki GUI or the java code
    arg_list = ["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.DBSCAN",
                     "-dbc.in", DATA_FILE_NAME,
                     "-parser.colsep", ",",
                     "-dbscan.epsilon", str(epsilon),
                     "-dbscan.minpts", str(minpts),
                    ]
    predicted_labels = run_elki_clustering(arg_list, X)
    return predicted_labels

def run_elki_kmeans(X, k):
    # You can read of the names of the parameters for the algorithm from the elki GUI or the java code
    arg_list = ["java", "-cp", ELKI_JAR, "de.lmu.ifi.dbs.elki.application.KDDCLIApplication",
                     "-algorithm", "clustering.kmeans.KMeansLloyd",
                     "-dbc.in", DATA_FILE_NAME,
                     "-parser.colsep", ",",
                     "-kmeans.k", str(k),
                    ]
    predicted_labels = run_elki_clustering(arg_list, X)
    return predicted_labels

def normalize(X):
   scaler = MinMaxScaler()
   return scaler.fit_transform(X)

if __name__ == "__main__":
    # Some random test data, load here the dataset that you want to use
  
    
    X, y = make_blobs(n_samples=100,
                      n_features=3,
                      centers=3,
                      cluster_std=0.1,
                      random_state=123
                      )
    """
    # Load Data
    data = np.genfromtxt('scale-d-10d.csv', delimiter=' ')
    X = data[:300,:10]

    labels = np.genfromtxt('scale-d-10d.csv', delimiter=' ',dtype="|U5")
    y = labels[:300,10]
    for i in range (y.shape[0]):
       y[i] = float(y[i][1:])

   

    
    test_data = np.array ([[0.15, 0.30, 0.00, 0.80], #c1
                        [0.12, 0.29, 0.35, 0.69], #c1
                        [0.00, 0.45, 0.61, 0.27], #c1
                        [0.39, 0.00, 1.00, 0.10], #c2
                        [0.59, 0.80, 0.10, 1.00], #c2
                        [0.54, 0.90, 0.29, 0.63], #c2           
                        [0.69, 1.00, 0.59, 0.28],
                        [1.00, 0.60, 0.81, 0.00],
                        [0.20, 0.27, 0.39, 0.97], #c1
                        [0.03, 0.47, 0.08, 0.57]])#c1
   
    X = test_data


    #X = np.genfromtxt('housing.csv')
    #X = normalize (X)
    #X = X[:200, :]
    #y = np.zeros_like (X[:,0])
    #y = np.zeros (506)
    
    #print (X)
    """
    print("Run elki")
    #pred = run_elki_kmeans(X, k=2)
    #pred = run_elki_dbscan(X, epsilon=0.5, minpts=2)
    pred = run_elki_P3C (X, poisson=1e-10)
    #print ("pred", pred.shape)
    print("pred_unique: ", set(pred.tolist()))
    print("Count: ", Counter(pred.tolist()))
    np.set_printoptions(threshold=np.inf)
    print("label:", pred)
    nmi = normalized_mutual_info_score(pred, y)
    print(f"NMI: {nmi:.4f}")
    pred_and_y = np.concatenate([pred[:,None],y[:,None]],axis=1)
    #print ("p+y:", pred_and_y.shape)
    np.savetxt("results_elki.csv", pred_and_y, delimiter=",", fmt="%.1f")