
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_value(value_func, state_encoder):
    grid_length = 4
    encoded_states = state_encoder(np.arange(0, 16))
    values = value_func(encoded_states).reshape((4,4))

    # values = critic(states).reshape((grid_length, grid_length)).numpy()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(values, annot=True, cmap='flare', ax=ax)
    ax.set_title('Critic State Value Estimates')

    return fig

def plot_state_action_value(q_func):
    pass