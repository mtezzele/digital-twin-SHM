import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from pathlib import Path
from tensorflow import keras
keras.backend.clear_session()

import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import DBNInference

from plotters.plotter import Plot

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"

rnd.seed(40)
np.random.seed(40)

def normalize_cpd(matrix):
    # normalizes the cpd table so that each column of values sums to 1
    col_sums = matrix.sum(axis=0, keepdims=True)
    return matrix / col_sums

class GetSetup:
    """
    Get Setup class. Handles the exchange of input information between the 
    other classes.
        
    :cvar numpy.ndarray states_list: list of all the possible states.
    :cvar numpy.ndarray actions_list: list of all the possible actions.
    :cvar int n_classes: number of possible damage classes.
    :cvar int n_level_step: number of points resulting from the discretization 
        of damage level.
    :cvar numpy.ndarray conf_mat_dt: cpd associated to the digital state estimation.
    
    :param numpy.ndarray states_list: list of all the possible states.
    :param numpy.ndarray actions_list: list of all the possible actions.
    :param int n_classes: number of possible damage classes.
    :param int n_level_step: degradation step advancing the evolution of the
        physical state.
    :param numpy.ndarray conf_mat_dt: cpd associated to the digital state estimation.
    """
    def __init__(self,
                 states_list,
                 actions_list,
                 n_class,
                 n_level_step,
                 conf_mat_dt):
        self.states_list = states_list
        self.actions_list = actions_list
        self.n_class = n_class
        self.n_level_step = n_level_step
        self.conf_mat_dt = conf_mat_dt
    
    def _get_low_diag_transition(self, prob_adv):
        """
        Returns a lower diagonal transition matrix under a first-order Markovian
        assumption and of a single state deterioration. It is assumed that damage
        may starts with any magnitude and grows with one-step increments. It is 
        also assumed an equal probability of advancement for the undamaged and 
        damaged configurations, without possibility of damage in multiple zones.
        
        :param float prob_adv: probability of advancement.
        :return: transtion matrix.
        :return type: numpy.ndarray.
        """
        n_states = self.states_list.shape[0]
        trans = np.zeros((n_states, n_states))
        # probability to stay in the undamaged condition
        trans[0, 0] = 1 - (prob_adv * (self.n_class-1))
        # probability to develop damage in an undamaged region
        #trans[1:, 0] = (prob_adv * (self.n_class-1))/(n_states-1)
        trans[1:n_states:self.n_level_step, 0] = prob_adv
        for i in range(1, n_states):
            if i % self.n_level_step == 0:
                trans[i, i] = 1.
            else:
                trans[i, i] = 1 - prob_adv
                trans[i+1, i] = prob_adv
        return trans

    def _get_restart_trainsition(self):
        """
        Returns a transition matrix for the perfect maintenace action.

        :return: transtion matrix.
        :return type: numpy.ndarray.
        """
        trans = np.zeros((self.states_list.shape[0], self.states_list.shape[0]))
        trans[0, :] = 1.
        return trans
        
    def _get_combined_transition(self, t1, t2):
        """
        Combines two cpds. This serves to combine cpds definining the conditioning
        of a common variable on different variables.
        
        :param numpy.ndarray t1: first cpd.
        :param numpy.ndarray t2: second cpd.
        :return: joint cpd.
        :return type: numpy.ndarray.
        """
        n_states = t1.shape[0]
        card_1 = t1.shape[1]
        card_2 = t2.shape[1]
        comb = np.zeros((n_states, card_1*card_2))
        for i in range(n_states):
            for j in range(card_1):
                for k in range(card_2):
                    comb[i, j*n_states+k] = t1[i, j] * t2[i, k]

        # normalizes the cpd table so that each column of values sums to 1
        return normalize_cpd(comb)

    def get_transitions(self,p_nothing,p_restricted):
        """
        Returns the transition matrix for each possible action. The transition 
        probabilities are an internal model of how the structural health is 
        expected to evolve, depending on which action is executed.
        
        :param float p_nothing: assumed probability to develop damage in each 
            region separately; after this, damage has the same probability to 
            advance but cannot can not spread over other regions.
        :param float p_restricted: in the same spirit of "p_nothing" this is the assumed 
            probability to develop damage in each region separately and subsequent
            advancement, yet in the case of restricted operational conditions.
        :return: concatenation of the transition matrices for each possible action.
        :return type: numpy.ndarray.
        """
        
        # p(D_t | D_t-1, U_t-1=do_nothing):
        trans_prob_nothing = self._get_low_diag_transition(prob_adv=p_nothing)
    
        # p(D_t | D_t-1, U_t-1=perfect_repair):
        trans_prob_perfect = self._get_restart_trainsition()
        
        # p(D_t | D_t-1, U_t-1=do_nothing):
        trans_prob_restrcited = self._get_low_diag_transition(prob_adv=p_restricted)
        
        return np.stack((trans_prob_nothing,
                         trans_prob_perfect,
                         trans_prob_restrcited), 0)
    
    def get_graph(self, cpd_d_to_u, transitions):
        """
        Returns all the structures required to create the main graph and the 
        prediction subgraph, and the associated cpds.
        
        :param numpy.ndarray cpd_d_to_u: cpd encoding a deterministic policy.
        :param transitions: stack of the transition matrices for each action.
        :return: nodes and edges defining the main graph;
                 nodes and edges defining the prediction subgraph;
                 cpds associated to the main graph;
                 cpds associated to the prediction subgraph.
        :return type: list; list; list; list.
        """
        states = self.states_list
        Ns = states.shape[0]
        actions = self.actions_list
        Na = actions.shape[0]
        
        # nodes and edges defining the graph
        # e.g. ((node_1, time_slice(node_1)), (node_2, timeslice(node_2)))
        graph_structure = ([(('U^{A}_{-1}', 0), ('D', 0)),
                            (('D_{NN}', 0), ('D', 0)),
                            (('D', 0), ('U^{P}', 0)),
                            (('D', 0), ('D', 1)),
                            (('U^{A}_{-1}', 1), ('D', 1)),
                            (('D_{NN}', 1), ('D', 1)),
                            (('D', 1), ('U^{P}', 1)),])
        
        # nodes and edges defining the prediction subgraph
        digital_subgraph = ([(('D', 0), ('U', 0)),
                             (('D', 0), ('D', 1)), 
                             (('U', 0), ('D', 1)),
                             (('D', 1), ('U', 1))])
        
        # p(D_t | D_t-1, U_t-1=do_nothing)
        cond_d_nothing = self._get_combined_transition(t1=self.conf_mat_dt,t2=transitions[0])
    
        # p(D_t | D_t-1, U_t-1=perfect_repair)
        cond_d_perfect = self._get_combined_transition(t1=self.conf_mat_dt,t2=transitions[1])
        
        # p(D_t | D_t-1, U_t-1=restrict_operations)
        cond_d_restricted = self._get_combined_transition(t1=self.conf_mat_dt,t2=transitions[2])
        
        cond_d = np.concatenate((cond_d_nothing,
                                 cond_d_perfect,
                                 cond_d_restricted), 1)
        
        # conditional probability tables to be associated with the model. Each CPD
        # should be an instance of `TabularCPD`. Note that while adding variables and
        # the evidence, nodes have to be of the form (node_name, time_slice)
        U_a_0_cpd = TabularCPD(('U^{A}_{-1}', 0),
                               Na,
                               np.ones((Na, 1)),
                               state_names={('U^{A}_{-1}', 0): actions.tolist()})
                
        D_NN_0_cpd = TabularCPD(('D_{NN}', 0),
                                Ns,
                                np.ones((Ns, 1)),
                                state_names = {('D_{NN}', 0): states.tolist()})
    
        D_0_cpd = TabularCPD(('D', 0),
                             Ns,
                             np.concatenate((self.conf_mat_dt,
                                             transitions[1],
                                             self.conf_mat_dt),1),
                             evidence=[('U^{A}_{-1}', 0),('D_{NN}', 0)],
                             evidence_card=[Na, Ns],
                             state_names={('D', 0): states.tolist(),
                                          ('U^{A}_{-1}', 0): actions.tolist(),
                                          ('D_{NN}', 0): states.tolist()})
        
        U_p_0_cpd = TabularCPD(('U^{P}', 0),
                               Na,
                               cpd_d_to_u,
                               evidence=[('D', 0)],
                               evidence_card=[Ns],
                               state_names={('U^{P}', 0): actions.tolist(),
                                            ('D', 0): states.tolist()})
    
        D_0_sub_cpd = TabularCPD(('D', 0),
                                 Ns,
                                 np.ones((Ns, 1)), 
                                 state_names={('D', 0): states.tolist()})
    
        U_0_sub_cpd = TabularCPD(('U', 0),
                                 Na,
                                 cpd_d_to_u,
                                 evidence=[('D', 0)],
                                 evidence_card=[Ns],
                                 state_names={('U', 0): actions.tolist(),
                                              ('D', 0): states.tolist()})
        
        U_a_1_cpd = TabularCPD(('U^{A}_{-1}', 1),
                               Na,
                               np.ones((Na, 1)),
                               state_names={('U^{A}_{-1}', 1): actions.tolist()})
    
        D_NN_1_cpd = TabularCPD(('D_{NN}', 1),
                                Ns,
                                np.ones((Ns, 1)),
                                state_names = {('D_{NN}', 1): states.tolist()})
    
        D_1_cpd = TabularCPD(('D', 1),
                             Ns,
                             cond_d,
                             evidence=[('U^{A}_{-1}', 1),('D_{NN}', 1),('D', 0)],
                             evidence_card=[Na, 
                                            Ns,
                                            Ns],
                             state_names={('D', 1): states.tolist(),
                                          ('U^{A}_{-1}', 1): actions.tolist(),
                                          ('D_{NN}', 1): states.tolist(),
                                          ('D', 0): states.tolist()})
        
        U_p_1_cpd = TabularCPD(('U^{P}', 1),
                               Na,
                               cpd_d_to_u,
                               evidence=[('D', 1)],
                               evidence_card=[Ns],
                               state_names={('U^{P}', 1): actions.tolist(),
                                            ('D', 1): states.tolist()})
    
        D_1_sub_cpd = TabularCPD(('D', 1),
                                 Ns,
                                 np.concatenate(np.array([transitions[i] for i 
                                                          in range(Na)]),1),
                                 evidence=[('U', 0),('D', 0)],
                                 evidence_card=[Na, Ns],
                                 state_names={('D', 1): states.tolist(),
                                              ('U', 0): actions.tolist(),
                                              ('D', 0): states.tolist()})
        
        U_1_sub_cpd = TabularCPD(('U', 1),
                                 Na,
                                 cpd_d_to_u,
                                 evidence=[('D', 1)],
                                 evidence_card=[Ns],
                                 state_names={('U', 1): actions.tolist(),
                                              ('D', 1): states.tolist()})
    
        list_cpd_graph = [U_a_0_cpd, D_NN_0_cpd, D_0_cpd, U_p_0_cpd, 
                          U_a_1_cpd, D_NN_1_cpd, D_1_cpd, U_p_1_cpd]
        list_cpd_subgraph = [D_0_sub_cpd, U_0_sub_cpd, D_1_sub_cpd, U_1_sub_cpd]
        
        return graph_structure, digital_subgraph, list_cpd_graph, list_cpd_subgraph


         
if __name__ == '__main__':

    # number of classes in the dataset
    n_class = 7
    # damage level discretization adopted to describe different states, related 
    # to the same damage location but different damage severity
    d_level_discr = 0.1
    # number of points resulting from the discretization of damage level 
    n_level_step = 6
    # path containing the data to be loaded
    path_data = './data/'
    # paths containing the models to be loaded
    path_classifier = './models/Classifier_total/model'
    path_regressors = [
        f'./models/Level_Regressor_{i}/model' for i in range(1, n_class)
    ]
    # load the trained SHM models
    classifier = keras.models.load_model(Path(path_classifier))
    regressors = [keras.models.load_model(Path(path_regressors[i])) for i in range(n_class-1)]
    # load the statistics used to normalize the damage levels for each regressor
    statistics_level = np.load(Path(f'{path_data}statistics_level.npy'))
    # load the confusion matrix previously obtained by evaluating the capabilities
    # of the SHM models to assess the digital state: row index identifies the 
    # true state while the column index is the predicted one. It is p(D_t | O_t)
    conf_mat_dt = np.load(Path(f'{path_data}confusion_digital_estimate.npy'))
    # add small quantity to avoid zero division during normalization
    conf_mat_dt = conf_mat_dt + 1e-5
    # normalizes the cpd table so that each column of values sums to 1
    conf_mat_dt = normalize_cpd(conf_mat_dt)

    # array of all the possible states
    possible_states = np.array(["0;0.0"])
    for i in range(1,n_class):
        for j in range(n_level_step):
            l = np.around(j*d_level_discr + 0.3, n_level_step-1)
            possible_states = np.append(
                possible_states, np.array([f"{str(i)};{str(l)}"]), 0
            )

    # array of all the possible actions
    possible_actions = np.array(["do_nothing", "perfect_repair", "restrict_operations" ])
    #"inspection"?

    ##########################################################################

    setup_bridge = GetSetup(states_list=possible_states,
                           actions_list=possible_actions,
                           n_class=n_class,
                           n_level_step=n_level_step,
                           conf_mat_dt=conf_mat_dt)

    transitions = setup_bridge.get_transitions(p_nothing=0.10,
                                               p_restricted=0.03)
