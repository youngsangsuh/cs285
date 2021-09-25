import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline

figsize=(5.7, 3)
export_dir = os.path.join('soln_pdf', 'figures')

sns.set_theme()
sns.set_context("paper")

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
    return X, Y


def read_q1_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'CartPole-v0' in split and batch in split:
            config_list = split[split.index(batch):split.index('CartPole-v0')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data


def read_q2_data():
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'InvertedPendulum-v2' in split:
            config_list = split[split.index('pg')+2:split.index('InvertedPendulum-v2')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data


def read_q3_data():
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        print(folder)
        split = folder.split('_')
        # NOTE(youngsang): change this
        if folder == 'q2_pg_q3_b40000_r0.005_LunarLanderContinuous-v2_19-09-2021_09-53-58':
            config_list = split[split.index('q3')+1:split.index('LunarLanderContinuous-v2')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data


def read_q4_data():
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'HalfCheetah-v2' in split and 'search' in split:
            config_list = split[split.index('search')+1:split.index('rtg')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()

            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data


def read_q4_optimal_data():
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'HalfCheetah-v2' in split and 'search' not in split:
            config_list = split[split.index('q4')+1:split.index('HalfCheetah-v2')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()

            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data


def read_q5_data():
    full_data = pd.DataFrame()

    for folder in os.listdir('data'):
        split = folder.split('_')
        if 'Hopper-v2' in split:
            config_list = split[split.index('r0.001')+1:split.index('Hopper-v2')]
            # print('_'.join(config))
            config = '_'.join(config_list)

            logdir = os.path.join('data', folder, 'events*')
            eventfile = glob.glob(logdir)[0]

            X, Y = get_section_results(eventfile)
            data = pd.DataFrame({'Iteration': range(len(X)), 
                                 'Config': np.repeat(config, len(X)), 
                                 'Train_EnvstepsSoFar': X, 
                                 'Eval_AverageReturn': Y})
            data['Eval_AverageReturn_Smooth'] = data['Eval_AverageReturn'].ewm(alpha=0.6).mean()

            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
        
    return full_data

#################################################
#################################################

# data_lb = read_q1_data('lb')
# data_lb

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_lb, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q1_lb.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_sb = read_q1_data('sb')
# data_sb

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_sb, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q1_sb.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_q2 = read_q2_data()
# print(data_q2)

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_q2, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q2.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_q3 = read_q3_data()
# print(data_q3)

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_q3, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config', ci=None)
# # plt.plot(data_q3['Iteration'], data_q3['Eval_AverageReturn_Smooth'], label='b40000_r0.005')
# # plt.xlabel('Iteration')
# # plt.ylabel('Eval_AverageReturn_Smooth')
# # plt.title('LunarLanderContinuous-v2')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q3.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_q4 = read_q4_data()
# print(data_q4)

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_q4, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q4_search.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_q4_optimal = read_q4_optimal_data()
# data_q4_optimal

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_q4_optimal, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q4_optimal.pdf'), bbox_inches='tight')

#################################################
#################################################

# data_q5 = read_q5_data()
# print(data_q5)

# plt.figure(figsize=figsize)
# sns.lineplot(data=data_q5, x='Iteration', y='Eval_AverageReturn_Smooth', hue='Config')
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.savefig(os.path.join(export_dir, 'q5.pdf'), bbox_inches='tight')
