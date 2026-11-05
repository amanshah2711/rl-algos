
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
import numpy as np
import gymnasium as gym

landmarks = {0 : (0,0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}

def generate_random_initial_state():
    taxi_pos = np.random.randint(0, 25)
    passenger_pos = np.random.randint(0, 4)
    while (destination := np.random.randint(0, 4)) == passenger_pos:
        pass
    state = (taxi_pos * 5 + passenger_pos) * 4 + destination
    return state

def get_image(state):
    env = gym.make(id='Taxi-v4', render_mode='rgb_array')
    obs, info = env.reset()
    env.unwrapped.s = state
    frame = env.render()
    return frame

def decode_state(state):
    destination = state % 4
    state = (state - destination) // 4

    passenger_pos = state % 5
    taxi_pos = (state-passenger_pos) // 5

    return taxi_pos, passenger_pos, destination

def encode_state(taxi_pos, passenger_pos, destination):
    return (taxi_pos * 5 + passenger_pos) * 4 + destination

def plot_action_on_image(q_func):
    initial_state = generate_random_initial_state()
    taxi_pos, passenger_pos, destination = decode_state(initial_state)
    passenger_row, passenger_col = landmarks[passenger_pos]

    encoded_initial_state = np.eye(500)[np.array([encode_state(pos, passenger_pos, destination) for pos in range(25)])]
    greedy_actions_pre = np.argmax(q_func(encoded_initial_state), axis=-1) # greedy policy induced by the q function

    encoded_post_pickup_state = np.eye(500)[np.array([encode_state(pos, 4, destination) for pos in range(25)])]
    greedy_actions_post = np.argmax(q_func(encoded_post_pickup_state), axis=-1)

    symbol_map = {0 : '\u2193', 1 : '\u2191', 2: '\u2192', 3 : '\u2190', 4:"P", 5:"D"}
    vectorized_map = np.vectorize(lambda x: symbol_map[x])
    symbols_pre = vectorized_map(greedy_actions_pre).reshape((5,5))
    symbols_post = vectorized_map(greedy_actions_post).reshape((5,5))

    frame1 = get_image(initial_state)
    frame2 = get_image(encode_state(passenger_row * 5 + passenger_col, 4, destination))
    fig_pre = plt.figure()
    ax_pre = fig_pre.add_subplot(1, 1, 1)
    ax_pre.set_title('Policy Overlay')
    ax_pre.axis('off')
    ax_pre.imshow(frame1)
    
    fig_post = plt.figure()
    ax_post = fig_post.add_subplot(1, 1, 1)
    ax_post.set_title('Policy Overlay')
    ax_post.axis('off')
    ax_post.imshow(frame2)

    for i in range(5):
        for j in range(5):
            x_coord = 48 + 44 + j * 88 # values for offsets and stride found online 
            y_coord = 128 + 28 + i * 56
            ax_pre.text(
                x_coord,
                y_coord,
                symbols_pre[i][j],
                fontsize=20,
                weight='bold',
                ha='center',
                va='center'
            )
            ax_post.text(
                x_coord,
                y_coord,
                symbols_post[i][j],
                fontsize=20,
                weight='bold',
                ha='center',
                va='center'
            )
    
    return fig_pre, fig_post

def plot_state_action_value(q_func):
    initial_state = generate_random_initial_state()
    taxi_pos, passenger_pos, destination = decode_state(initial_state)
    passenger_row, passenger_col = landmarks[passenger_pos]
    destination_row, destination_col = landmarks[destination]

    encoded_initial_state = np.eye(500)[np.array([encode_state(pos, passenger_pos, destination) for pos in range(25)])]
    greedy_actions_pre = np.argmax(q_func(encoded_initial_state), axis=-1) # greedy policy induced by the q function
    q_values_pre = np.max(q_func(encoded_initial_state), axis=-1).reshape((5,5))

    encoded_post_pickup_state = np.eye(500)[np.array([encode_state(pos, 4, destination) for pos in range(25)])]
    greedy_actions_post = np.argmax(q_func(encoded_post_pickup_state), axis=-1)
    q_values_post = np.max(q_func(encoded_post_pickup_state), axis=-1).reshape((5,5))

    symbol_map = {0 : '\u2193', 1 : '\u2191', 2: '\u2192', 3 : '\u2190', 4:"P", 5:"D"}
    vectorized_map = np.vectorize(lambda x: symbol_map[x])
    symbols_pre = vectorized_map(greedy_actions_pre).reshape((5,5))
    symbols_post = vectorized_map(greedy_actions_post).reshape((5,5))

    fig_pre = plt.figure()
    pre = fig_pre.add_subplot(1, 1, 1)
    pre.scatter(
        passenger_col + 0.25,
        passenger_row + 0.25,
        marker='o',
        s=150,
        color='deepskyblue',
        edgecolors='black',
        zorder=10
    )
    pre.scatter(
        destination_col + 0.25,
        destination_row + 0.25,
        marker='*',
        s=200,
        color='gold',
        edgecolors='black',
        zorder=10
    )
    pre.set_title('State-Action Values Before Pickup')

    fig_post = plt.figure()
    post = fig_post.add_subplot(1,1,1)
    post.scatter(
        destination_col + 0.25,
        destination_row + 0.25,
        marker='*',
        s=200,
        color='gold',
        edgecolors='black',
        zorder=10
    )
    post.set_title('State-Action Values Post Pickup')

    sns.heatmap(q_values_pre, annot=symbols_pre, fmt='', cmap='flare', ax=pre, annot_kws={"size": 16})
    sns.heatmap(q_values_post, annot=symbols_post, fmt='', cmap='flare', ax=post, annot_kws={"size": 16})

    return fig_pre, fig_post