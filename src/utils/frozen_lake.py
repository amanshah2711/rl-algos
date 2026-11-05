import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_value(value_func, length, height): # Assumes one-hot encoding is used by models
    states = np.arange(0, length * height) 
    encoded_states = np.eye(length * height)[states] # Shape : (num_possible_states, obs_dim)
    values = value_func(encoded_states).reshape((height, length))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax = sns.heatmap(values, annot=True, cmap='flare', ax=ax)
    ax.set_title('Critic State Value Estimates')

    return fig

def plot_state_action_value(q_func, length, height): # assumes one-hot encoding and interface with q_func is q_func(states) := list of action values
    states = np.arange(0, length * height)
    encoded_states = np.eye(length * height)[states] # Shape : (num_possible_states, obs_dim)
    values = np.max(q_func(encoded_states), axis=-1).reshape((height, length))
    greedy_actions = np.argmax(q_func(encoded_states), axis=-1).reshape((height, length))

    x, y = np.meshgrid(
        np.arange(length) + 0.5,
        np.arange(height) + 0.5
    )

    dx = np.array([-1, 0, 1, 0])[greedy_actions]
    dy = np.array([0, 1, 0, -1])[greedy_actions]

    fig, ax = plt.subplots()
    sns.heatmap(values, annot=False, cmap='flare', ax=ax)
    ax.set_title('Critic State Value Estimates')

    ax.quiver(
        x, y, dx, dy,
        color='white',
        angles='xy',
        scale_units='xy',
        pivot='middle',
    )

    return fig