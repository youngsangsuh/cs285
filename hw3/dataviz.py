import os
import tensorflow as tf
import glob
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

figsize=(5.7, 3)
export_dir = os.path.join('figures')
data_dir = os.path.join('data')

sns.set_theme()
sns.set_context("paper")

def get_section_results(file, *tags):
    """
        requires tensorflow==1.12.0
    """
    data_dict = {tag: [] for tag in tags}
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag in data_dict:
                data_dict[v.tag].append(v.simple_value)
    data_dict = {tag: np.array(data_dict[tag]) for tag in data_dict}
    return data_dict

def read_q1_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'MsPacman-v0' in split and batch in split:
            config_list = split[split.index(batch):split.index('MsPacman-v0')+1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Train_AverageReturn', 'Train_BestReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

q1_data = read_q1_data('q1')
q1_longform = q1_data.drop('Config', axis=1).melt(id_vars=['Train_EnvstepsSoFar'], value_vars=['Train_AverageReturn', 'Train_BestReturn'])
q1_longform

plt.figure(figsize=figsize)
ax = sns.lineplot(data=q1_longform, x='Train_EnvstepsSoFar', y='value', hue='variable')
ax.set(xlabel='Training Steps', ylabel='Reward')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.savefig(os.path.join(export_dir, 'q1.pdf'), bbox_inches='tight')

def read_q2_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'LunarLander-v3' in split and batch in split:
            config_list = split[split.index(batch)+1:split.index('LunarLander-v3')-1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            

            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                    'Config': np.repeat(config, len(data_dict[idx])), 
                                    **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)
    # full_data.loc[:, 'Train_AverageReturn'] /= 3
    return full_data

full_q2_data = read_q2_data('q2')
full_q2_data


plt.figure(figsize=figsize)
sns.lineplot(data=full_q2_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn', hue='Config')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig(os.path.join(export_dir, 'q2.pdf'), bbox_inches='tight')



def read_q3_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if '02-54-53' in split:
            config_list = split[split.index('q2')+1:split.index('LunarLander-v3')-1]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

        if all([tag in split for tag in batch]):
            config_list = split[split.index('q3')+1:split.index('LunarLander-v3')]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Train_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

full_q3_data = read_q3_data(['q3', 'LunarLander-v3'])
full_q3_data

plt.figure(figsize=figsize)
sns.lineplot(data=full_q3_data, x='Train_EnvstepsSoFar', y='Train_AverageReturn', hue='Config')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig(os.path.join(export_dir, 'q3.pdf'), bbox_inches='tight')

def read_q4_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'CartPole-v0' in split and batch in split:
            config_list = split[split.index(batch)+1:split.index('CartPole-v0')]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Eval_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

full_q4_data = read_q4_data('q4')
full_q4_data

plt.figure(figsize=figsize)
sns.lineplot(data=full_q4_data, x='Train_EnvstepsSoFar', y='Eval_AverageReturn', hue='Config')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig(os.path.join(export_dir, 'q4.pdf'), bbox_inches='tight')


def inverted_pendulum_q5_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'InvertedPendulum-v2' in split and batch in split:
            config_list = split[split.index(batch)+1:split.index('InvertedPendulum-v2')]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Eval_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

inverted_pendulum_q5_data = inverted_pendulum_q5_data('q5')
inverted_pendulum_q5_data


plt.figure(figsize=figsize)
sns.lineplot(data=inverted_pendulum_q5_data, x='Train_EnvstepsSoFar', y='Eval_AverageReturn')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig(os.path.join(export_dir, 'q5_inverted_pendulum.pdf'), bbox_inches='tight')

def half_cheetah_q5_data(batch):
    full_data = pd.DataFrame()

    for folder in os.listdir(data_dir):
        split = [s.strip() for s in folder.split('_')]
        if 'HalfCheetah-v2' in split and batch in split:
            config_list = split[split.index(batch)+1:split.index('HalfCheetah-v2')]
            config = '_'.join(config_list)

            logdir = os.path.join(data_dir, folder, 'events*')
            print(logdir)
            eventfile = glob.glob(logdir)[0]

            data_dict = get_section_results(eventfile, 'Train_EnvstepsSoFar', 'Eval_AverageReturn')
            idx = 'Train_EnvstepsSoFar'
            for key in data_dict:
                while len(data_dict[key]) < len(data_dict[idx]):
                    data_dict[key] = np.insert(data_dict[key], 0, None)
            
            data = pd.DataFrame({'Iteration': range(len(data_dict[idx])), 
                                 'Config': np.repeat(config, len(data_dict[idx])), 
                                 **data_dict})
            full_data = pd.concat([full_data, data], axis=0, ignore_index=True)

    return full_data

half_cheetah_q5_data = half_cheetah_q5_data('q5')
half_cheetah_q5_data

plt.figure(figsize=figsize)
sns.lineplot(data=half_cheetah_q5_data, x='Train_EnvstepsSoFar', y='Eval_AverageReturn')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig(os.path.join(export_dir, 'q5_half_cheetah.pdf'), bbox_inches='tight')
