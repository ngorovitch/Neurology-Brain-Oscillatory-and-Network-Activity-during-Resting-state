#%% importing routine libraries
import pyedflib
import numpy as np
import connectivipy as cp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
import re

#%% 1-Connectivity graph
class EEGGraph():
    '''Container object representing the electroencephalography(EEG) brain activity time series data as a graph'''
    def __init__(self, path):
        Data = self.from_edf(path)
        self.number_of_signals = Data['number of signals']
        self.signals = Data['signals']
        self.labels = [re.sub('[^\w]',"", l) for l in Data['labels']]

        
    def from_edf(self, path):
        '''
        Function to read data from edf file
        '''
        f = pyedflib.EdfReader(path)
        n = f.signals_in_file
        signal_labels = f.getSignalLabels()
        sigbufs = np.zeros((n, f.getNSamples()[0]))
        for i in np.arange(n):
            sigbufs[i, :] = f.readSignal(i)
        f._close(); del f
        return {'number of signals': n, 'signals': sigbufs, 'labels': signal_labels}
    
    
    def plot_data(self, nbr_of_channels = 10):
        sampling_frequency = 160.0
        data = cp.Data(self.signals[:nbr_of_channels, :], sampling_frequency, self.labels[:nbr_of_channels])
        data.plot_data()

    def get_connectivity_matrix_PDC(self):
        ''' '''
        # assign static class cp.Mvar to variable mv
        mv = cp.Mvar
        # find best model order using Vieira-Morf algorithm        
        max_model_order = 20
        method = 'vm'
        best_order, crit = mv.order_akaike(self.signals, max_model_order, method)
        # now let's fit parameters to the signal
        av, vf = mv.fit(self.signals, best_order, method)        
        # now we can calculate Partial Directed Coherence (PDC) from the data
        pdc = cp.conn.PDC()
        sampling_frequency = 160.0
        pdcval = pdc.calculate(Acoef = av, Vcoef = vf, fs = sampling_frequency, resolution=100)
        # result = pdc.significance(data = self.signals, method = 'yw', Nrep = 20, alpha = 0.2)
        self.pdc_matrix = pdcval
        return pdcval
        
    def get_connectivity_matrix_DTF(self):
        ''' '''
        # assign static class cp.Mvar to variable mv
        mv = cp.Mvar
        # find best model order using Vieira-Morf algorithm        
        max_model_order = 20
        method = 'vm'
        best_order, crit = mv.order_akaike(self.signals, max_model_order, method)
        # now let's fit parameters to the signal
        av, vf = mv.fit(self.signals, best_order, method)        
        # now we can calculate Partial Directed Coherence (PDC) from the data
        dtf = cp.conn.DTF()
        sampling_frequency = 160.0
        dtfval = dtf.calculate(Acoef = av, Vcoef = vf, fs = sampling_frequency, resolution=100)
        # result = dtf.significance(data = self.signals, method = 'yw', Nrep = 20, alpha = 0.2)
        self.dtf_matrix = dtfval
        return dtfval
    
    
    def __get_adjacency_matrix(self, tresh, data, weighted = False):
        ''' '''
        N = data.shape[0]
        A = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    if data[i, j] > tresh:
                        if weighted:
                            A[i, j] = data[i, j]
                        else:
                            A[i, j] = 1
        return A
    
    
    def get_binary_adjacency_matrix(self, data, relevant_frequency, mvar_estimator = 'pdc', target_density = 20):
        ''' 
        This function returns a binary connectivity matrix with network density equal to "target_density"
        Inputs:
            data: a non binary connectivity matrix (p*k*k), p is the resolution, k is the number of channels
            relevant_frequency: an integer, it is the chosen frequency (chosen resolution)
            target_density: an integer, it is the disered density for the output binary matrix
        Outputs:
            A: a binary matrix (k*k), k is the number of channels
            G: a networkx graph deriving from A
            threshold: a float value
        '''
        # Predisposing a list of threshold ranging from 0 to 1
        threshold_range = np.arange(0, 1, 0.001)
        # unweighted graph loop
        for threshold1 in threshold_range:
            # Get the edges and the current binary matrix obtained using the current threshold
            A = self.__get_adjacency_matrix(threshold1, data[relevant_frequency,:,:], weighted = False)
            # Generate a graph from the current binary matrix in order to check it's density
            G = nx.from_numpy_matrix(A, create_using=nx.DiGraph)
            density = nx.density(G)
            # We stop iterating whenever we obtain a graph with density equal to the target density
            if int(density*100) == target_density:
                break
        # weighted graph loop
        for threshold2 in threshold_range:
            A_w = self.__get_adjacency_matrix(threshold2, data[relevant_frequency,:,:], weighted = True)
            # Generate a graph from the current adjacency matrix in order to check it's density
            G_w = nx.from_numpy_matrix(A_w, create_using = nx.DiGraph)
            density = nx.density(G_w)
            # We stop iterating whenever we obtain a graph with density equal to the target density
            if int(density*100) == target_density:
                break
            
        if mvar_estimator == 'pdc':
            self.bin_adj_matrix_pdc = A
            self.adj_matrix_pdc = A_w
            self.pdc_graph = G
            self.pdc_weighted_graph = G_w
        elif mvar_estimator == 'dtf':
            self.bin_adj_matrix_dtf = A
            self.adj_matrix_dtf = A_w
            self.dtf_graph = G
            self.dtf_weighted_graph = G_w
        else:
            raise ValueError("Invalid value for parameter mvar_estimator: should be 'pdc' or 'dtf'") 
        return A, G, threshold1
    
    def __get_positions(self, channel_names, locations):
        return {i: locations[cn] for i, cn in enumerate(channel_names)}
    
    def show_graph(self, G, title, locations, saving_path = 'graph', save = True):
        ''' '''
        plt.figure(figsize=(15,10))
        nodes = np.arange(len(self.labels))
        labels = dict(zip(nodes, self.labels))
        plt.title(title)
        nx.draw(G, node_size=800, labels=labels, with_labels=True, pos = self.__get_positions(self.labels, locations))
        if save: plt.savefig(saving_path, bbox_inches = 'tight')
        plt.show()
        
    def draw_local_indices(self, G, title, locations, measures, saving_path = 'graph', save = True):
        ''' '''
        # specify a custom
        # formatter for colorbar labels
        # in return select desired format
        def myfmt(x, pos):
            return '{0:.2f}'.format(x)
        
        plt.figure(figsize=(15,10))
        pos = self.__get_positions(self.labels, locations)
        nodes = nx.draw_networkx_nodes(G, pos, node_size=800, cmap=plt.cm.plasma, 
                                       node_color=list(measures.values()),
                                       nodelist=list(measures.keys()))
        nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
        node_list = np.arange(len(self.labels))
        labels = dict(zip(node_list, self.labels))
        nx.draw_networkx_labels(G, pos, labels=labels)
        nx.draw_networkx_edges(G, pos)
        plt.title(title)
        plt.colorbar(nodes, format=ticker.FuncFormatter(myfmt))
        plt.axis('off')
        if save: plt.savefig(saving_path, bbox_inches = 'tight')
        plt.show()


#%% 2-Graph theory indices
        
    
    
#%% End.