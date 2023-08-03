import os

from pathlib import Path
from tensorflow import keras
keras.backend.clear_session()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap

class Plot:
    
    def __init__(self, graph,path):
        self.graph=graph    
        self.path=path
        if not os.path.exists(Path(self.path)):
            os.mkdir(Path(self.path))
            
        
    def plot_history_all_together(self, figsize=(10, 7)):

        viridis_big = mpl.colormaps['Blues']
        newcmp = ListedColormap(viridis_big(np.linspace(0.1, 0.7, 128)))
        
        ############ history D ############
        
        hist = self.graph.hist_var
        hist_prob_D = self.graph.hist_d_state_prob.to_numpy()
        n_delta_steps = hist_prob_D.shape[1]//(self.graph.physical_asset.n_classes-1)

        index_estimated = np.array([np.where(self.graph.states_list == 
                            hist.d_state.to_numpy()[i])[0] for i in 
                            range(hist_prob_D.shape[0])])

        # non discretized
        index_truth = np.array([np.max([0.,(1+(int(hist.p_state.to_numpy()[i].split(';')[0])-1) * 6 \
                                + (float(hist.p_state.to_numpy()[i].split(';')[1])-0.3)/0.1)]) for i in \
                                range(hist_prob_D.shape[0])])

        index_pad_est = np.zeros((self.graph.physical_asset.n_classes-1,
                                hist_prob_D.shape[0]))
        index_pad_tru = np.zeros((self.graph.physical_asset.n_classes-1,
                                hist_prob_D.shape[0]))
        #n_regions x n_step x n_intervals
        hist_prob_pad = np.zeros((self.graph.physical_asset.n_classes-1,
                                hist_prob_D.shape[0],1+n_delta_steps))

        hist_prob_pad[:,:,0] = hist_prob_D[:,0]
        for i in range(1,hist_prob_D.shape[1]):
            zone = (i-1)//n_delta_steps
            hist_prob_pad[zone,:,i-(zone*n_delta_steps)] = hist_prob_D[:,i]

        for i in range(hist_prob_D.shape[0]):
            if index_estimated[i]!=0:
                zone = int((index_estimated[i]-1)//n_delta_steps)
                index_pad_est[zone,i] = index_estimated[i]-(zone*n_delta_steps)
            if int(hist.p_state.to_numpy()[i].split(';')[0])!=0:
                zone = int(hist.p_state.to_numpy()[i].split(';')[0])-1
                index_pad_tru[zone,i] = index_truth[i]-(zone*n_delta_steps)

        self.indices = [
            i for i, history in enumerate(index_pad_tru) if np.amax(history) > 0
        ]

        ############ history U ############
        
        hist_prob_U = self.graph.hist_actions_prob
        
        index_estimated = np.array([np.where(self.graph.actions_list == 
                            hist.current_action.to_numpy()[i])[0] for i in 
                            range(hist_prob_U.shape[0])])
        
        index_optimal = np.array([np.argmax(self.graph.planner.policy.T[:,np.where(
                    self.graph.states_list == hist.p_state_discrete.to_numpy()[i])[0]]) 
                    for i in range(hist_prob_U.shape[0])])

        plt.rc('font', size=19)
        fig, ax = plt.subplots(nrows=len(self.indices)+1, sharex=True, figsize=figsize,
                               height_ratios=len(self.indices)*[1] + [0.5])
        
        
        for i, j in enumerate(self.indices):
            img1 = ax[i].pcolormesh(np.arange(hist_prob_pad[j].shape[0]),
                                   np.arange(hist_prob_pad[j].shape[1]), 
                                   hist_prob_pad[j].T, 
                                   shading='nearest', vmin=0., vmax=1., cmap=newcmp)

            ax[i].plot(np.arange(0, hist_prob_pad[j].shape[0], 1), index_pad_est[j],
                     color='red', alpha=0.7, linewidth=1.8)
            ax[i].plot(np.arange(0, hist_prob_pad[j].shape[0], 1), index_pad_tru[j],
                     color='black', alpha=0.7, linestyle='dashed', linewidth=1.8)

            ax[i].grid(linestyle='dotted')
            ax[i].set_xticks(np.arange(+.5, hist_prob_pad[j].shape[0]-0.5, 1));
            ax[i].set_yticks(np.arange(+.5, hist_prob_pad[j].shape[1]-0.5, 1));
            ax[i].set_xticklabels(labels=[]);
            ax[i].set_yticklabels(labels=[]);

            ax[i].set_xticks(np.arange(0, hist_prob_pad[j].shape[0], 5), minor=True);
            ax[i].set_yticks(np.arange(0, hist_prob_pad[j].shape[1], 1), minor=True);
            ax[i].set_xticklabels(list(map(str, np.arange(0, hist_prob_pad[j].shape[0], 5))),
                                  minor=True, 
                                  fontsize=18);
            ax[i].set_yticklabels(['No dmg', '', '$40\%$','','$60\%$','','$80\%$'],
                                  minor=True,
                                  fontsize=18);

            ax[i].set_ylabel(ylabel=f'$\delta(\Omega_{str(j+1)})$', labelpad=-18)
            
        i=len(self.indices)
        
        img2 = ax[i].pcolormesh(np.arange(hist_prob_U.shape[0]),
                            np.arange(hist_prob_U.shape[1]), 
                            hist_prob_U.T, 
                            shading='nearest', vmin=0., vmax=1., cmap=newcmp)
    
        ax[i].plot(np.arange(0,hist_prob_U.shape[0],1),index_estimated,
                  color='red', alpha=0.7, linewidth=1.8)
        
        ax[i].plot(np.arange(0,hist_prob_U.shape[0],1),index_optimal,
                  color='black', alpha=0.7, linestyle='dashed', linewidth=1.8)    
            
        ax[i].grid(linestyle='dotted')
        ax[i].set_xticks(np.arange(+.5, hist_prob_U.shape[0]-0.5, 1));
        ax[i].set_yticks(np.arange(+.5, hist_prob_U.shape[1]-0.5, 1));
        ax[i].set_xticklabels(labels=[]);
        ax[i].set_yticklabels(labels=[]);
        
        ax[i].set_xticks(np.arange(0, hist_prob_U.shape[0], 5), minor=True);
        ax[i].set_yticks(np.arange(0, hist_prob_U.shape[1], 1), minor=True);
        ax[i].set_xticklabels(list(map(str, np.arange(0, hist_prob_U.shape[0], 5))),
                           minor=True, 
                           fontsize=18);
        
        ax[i].set_yticklabels(['DN', 'PM', 'RE'], minor=True, fontsize=18);
        ax[0].legend(['Digital twin', 'Ground truth'], ncol=1, loc='upper right')

        plt.ylabel(ylabel='Actions',labelpad=18);
        plt.xlabel('Time step $t$');
        
        cbar = fig.colorbar(img1, ax=ax[0:len(self.indices)], orientation='vertical', pad=0.03, aspect=30);
            
        cbar.set_label('$P \,(D_t | D_{t-1}, D^{\mathtt{NN}}_{t}, U^{\mathtt{A}}_{t-1}=u^{\mathtt{A}}_{t-1})$')

        cbar = fig.colorbar(img2, ax=ax[len(self.indices)], orientation='vertical', pad=0.03, aspect=7, ticks=[0.0, 0.5, 1.0])
        cbar.set_label('$P \, (U^{\mathtt{P}}_t|D_t)$')
        
        plt.savefig(Path(self.path + '/history.pdf'))


    def plot_prediction_all_together(self, n_steps=1, n_samples=1000, figsize=(10, 5)):
        viridis_big = mpl.colormaps['Blues']
        newcmp = ListedColormap(viridis_big(np.linspace(0.1, 0.7, 128)))

        predict_D, predict_U = self.graph.predict(n_steps=n_steps, n_samples=n_samples) 

        predict_D = predict_D.to_numpy()
        predict_U = predict_U.to_numpy()
        
        n_delta_steps = predict_D.shape[1]//(self.graph.physical_asset.n_classes-1)

        pred_prob_pad = np.zeros((self.graph.physical_asset.n_classes-1,
                                  predict_D.shape[0],1+n_delta_steps))

        pred_prob_pad[:,:,0] = predict_D[:,0]
        for i in range(1,predict_D.shape[1]):
            zone = (i-1)//n_delta_steps
            pred_prob_pad[zone,:,i-(zone*n_delta_steps)] = predict_D[:,i]
            
        plt.rc('font', size=19)
        fig, ax = plt.subplots(nrows=2, sharex=True, figsize=figsize, height_ratios=[1, 0.4])
        
        img = ax[0].pcolormesh(np.arange(pred_prob_pad[self.indices[-1]].shape[0]),
                               np.arange(pred_prob_pad[self.indices[-1]].shape[1]), 
                               pred_prob_pad[self.indices[-1]].T, 
                               shading='nearest', vmin=0., vmax=1., cmap=newcmp)

        ax[0].grid(linestyle='dotted')
        ax[0].set_xticks(np.arange(+.5, pred_prob_pad[5].shape[0]-0.5, 1));
        ax[0].set_yticks(np.arange(+.5, pred_prob_pad[5].shape[1]-0.5, 1));
        ax[0].set_xticklabels(labels=[]);
        ax[0].set_yticklabels(labels=[]);
    
        ax[0].set_xticks(np.arange(0, pred_prob_pad[5].shape[0], 5), minor=True);
        ax[0].set_yticks(np.arange(0, pred_prob_pad[5].shape[1], 1), minor=True);
        ax[0].set_xticklabels(list(map(str, np.arange(0, pred_prob_pad[5].shape[0], 5))), minor=True, fontsize=18);
        ax[0].set_yticklabels(['No dmg','$30\%$','$40\%$','$50\%$','$60\%$','$70\%$','$80\%$'], minor=True, fontsize=18);
            
        ax[0].set_ylabel(ylabel=f'$\delta(\Omega_{str(self.indices[-1]+1)})$', labelpad=-18)

        plt.xlabel('Time step $t$');
        
        cbar = fig.colorbar(img, ax=ax[0], orientation='vertical', pad=0.03, aspect=17)
        cbar.set_label('$P \, (D_t|D_{t-1},U^{\mathtt{P}}_{t-1})$')


        img2 = ax[1].pcolormesh(np.arange(predict_U.shape[0]),
                                np.arange(predict_U.shape[1]), 
                                predict_U.T, 
                                shading='nearest', vmin=0., vmax=1., cmap=newcmp)

        ax[1].grid(linestyle='dotted')
        ax[1].set_xticks(np.arange(+.5, predict_U.shape[0]-0.5, 1));
        ax[1].set_yticks(np.arange(+.5, predict_U.shape[1]-0.5, 1));
        ax[1].set_xticklabels(labels=[]);
        ax[1].set_yticklabels(labels=[]);

        ax[1].set_xticks(np.arange(0, predict_U.shape[0], 5), minor=True);
        ax[1].set_yticks(np.arange(0, predict_U.shape[1], 1), minor=True);
        ax[1].set_xticklabels(list(map(str, np.arange(0, predict_U.shape[0], 5))), minor=True, fontsize=18);
        
        ax[1].set_yticklabels(['DN', 'PM', 'RE'], minor=True, fontsize=18);

        plt.xlabel('Time step $t$');
        plt.ylabel('Actions', labelpad=18);

        cbar = fig.colorbar(img2, ax=ax[1], orientation='vertical', pad=0.03, aspect=7, ticks=[0.0, 0.5, 1.0])
        cbar.ax.set_yticklabels(['0.00', '0.50', '1.00'])
        cbar.set_label('$P \, (U^{\mathtt{P}}_t|D_t)$')

        plt.savefig(Path(self.path + '/predictions.pdf'))