import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

figsize=(5.7, 3)
export_dir = os.path.join('soln_pdf', 'figures')
data_dir = os.path.join('data')

sns.set_theme()
sns.set_context("paper")

def get_section_results(file, *tags):
    """
        requires tensorflow==1.12.0
    """
    data_dict = {tag: [] for tag in tags}
    for e in tf.compat.v1.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in data_dict:
                data_dict[v.tag].append(v.simple_value)
    data_dict = {tag: np.array(data_dict[tag]) for tag in data_dict}
    return data_dict


def read_q4_data(batches, tags):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        # print(split)
        if all([b in split or any([s.startswith(b) for s in split]) for b in batches]):
            config_list = split[split.index(batches[0]):-2]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, *tags)
            idx = 'Train_EnvstepsSoFar'
            
            '''
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)'''
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data


params = ['horizon', 'numseq', 'ensemble']
for p in params:
    q4_param_data = read_q4_data(['q4', p], ['Train_EnvstepsSoFar', 'Eval_AverageReturn'])
    q4_param_data.Config = q4_param_data.Config.apply(lambda x: x.split('_')[2])

    plt.figure(figsize=figsize)
    ax = sns.lineplot(data=q4_param_data, x='Train_EnvstepsSoFar', y='Eval_AverageReturn', hue='Config')
    ax.set(xlabel='Training Steps', ylabel='Eval_AverageReturn')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    plt.title(p)
    plt.savefig(os.path.join(export_dir, f'q4_{p}.pdf'), bbox_inches='tight')