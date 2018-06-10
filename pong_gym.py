import gym
import numpy as np
import _pickle as pickle
# import sys

def downsample(image):
    #half the pizels for some reason its ok for this game
    return image[::2, ::2,0]

def remove_color(image):
    #rbg to gray
    return image[:, :, 0]

def remove_background(image):
    #numbers specfici to this pong type
    #basic masking
    image[image == 144]  = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation):
    # makes the 210x160x3 uint8 frame into a 6400 float vector
    processed = input_observation[35:195]
    processed = processed[::2, ::2, 0]
    processed[processed == 144] = 0
    processed[processed == 109] == 0
    processed[processed != 0 ] = 1
    return processed.astype(np.float).ravel()

    # processed_observation = remove_color(processed_observation)
    # processed_observation = remove_background(processed_observation)
    # processed_observation[processed_observation != 0] = 1 #sets useful stuff to 1s
    # processed_observation = processed_observation.astype(np.float).ravel() #magically flattens from 80x80 to 1600x1

    # if prev_processed_observation is not None:
    #     input_observation = processed_observation - prev_processed_observation
    # else:
    #     input_observation = np.zeros(input_dimensions)
    # prev_processed_observations = processed_observation
    # return input_observation, prev_processed_observations

# Activations here
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def policy_forward(model, observation):
    #computes a hidden layer value and output layer
    hidden_layer_values = np.dot(model['W1'], observation) # first pass
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, model['W2']) # second pass
    decision = sigmoid(output_layer_values)
    return decision, hidden_layer_values

def choose_action(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        #up in gym, if our prob is bigger than the random action then yeah be good
        return 2
    else:
        #go down if not
        return 3

def policy_backward(model, hidden_layer_values, logged_gradient, observation):
    #based on the deep learning textbook 4 fund eqs
    dW2 = np.dot(hidden_layer_values.T, logged_gradient).ravel()
    delta_h = np.outer(logged_gradient, model['W2'])
    delta_h = relu(delta_h)
    dW1 = = np.dot(delta_h.T, observation)
    return {'W1': dW1, 'W2': dW2}

def discount_rewards(rewards, gamma):
    # 20 steps ago < 1 stepp ago
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    #discount gradient and normalizes rewards
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards

def update_weights(model, grad_buff, rms_cache, learning_rate, decay_rate):
    epsilon = 1e-5
    for k, v in model.iteritems():
        g = grad_buff[k]
        rms_cache[k] = decay_rate * rms_cache[k] + (1 - decay_rate) * g ** 2
        model[k] += (learning_rate * g)/(np.sqrt(rms_cache[k]) + epsilon)
        grad_buff[k] = np.zeros_like(v)


def main():


    env = gym.make("Pong-v0")
    observation = env.reset()
    new = True
    batch_size = 10 # episodes before shifting the weights
    gamma = 0.99 # discount
    decay_rate = 0.99
    num_hidden_layer_neurons = 200 # num neurons
    input_dimensions = 80 * 80 # image size
    learning_rate = 1e-4

    episode_number = 0
    reward_sum = 0
    render = False
    running_reward = None
    prev_observations = None

    # params for RMSProp, chcekc out the link: To be used with rmsprop algorithm (http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop)
    if new:
        model = {}
        model['W1'] = np.random.randn(num_hidden_layer_neurons, input_dimensions) / np.sqrt(input_dimensions)
        model['W2'] = np.random.randn(num_hidden_layer_neurons) / np.sqrt(num_hidden_layer_neurons)
    else:
        model = pickle.load(open('weights.pickle', 'rb'))

    grad_buffer = {k : np.zeros_like(v) for k, v in model.iteritems()}
    rmsprop_cache = {k : np.zeros_like(v) for k, v in model.iteritems()}

    hid_s, obs_s, gradlog, reward_list= [], [], [], []

    while True:
        if render: env.render()
        current_observations = preprocess_observations(observation)
        if prev_observations is not None:
            diff = current_observations - prev_observations
        else:
            diff = current_observations - np.zeros(input_dimensions)
        prev_observations = current_observations

        action_prob, hidden_state = policy_forward(model, diff)
        action = choose_action(action_prob)

        obs_s.append(diff)
        hid_s.append(hidden_state)

        y = 1 if action == 2 else 0
        gradlog.append(y - action_prob) #grad


        observation, reward, done, info = env.step(action)

        reward_sum += reward
        reward_list.append(reward)

        if done:
            episode_number += 1

            episode_obs = np.vstack(obs_s)
            episode_hid = np.vstack(hid_s)
            episode_gradlog = np.vstack(gradlog)
            episode_rewards = np.vstack(reward_list)
            hid_s, obs_s, reward_list, gradlog = [], [], [], []

            #next step takes in the gamma and applies the necessary transformations
            episode_gradlog_discounted = discount_with_rewards(episode_gradlog, episode_rewards, gamma)

            gradient = policy_backward(model, episode_hid, episode_gradlog_discounted)

            for k in model:
                grad_buffer[k] += gradient[k]

            if episode_number % batch_size == 0:
                for k, v in model.iteritems():
                    g = grad_buffer[k]
                    update_weights(model, grad_buffer, rmsprop_cache, decay_rate)

                # update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
                pickle_out = open("weights.pickle", "wb")
                # temp_dict = {
                #     'weight': weights,
                #     'g_squared': expectation_g_squared,
                #     'g_dict': g_dict,
                #     'running_mean': running_reward,
                #     'reward_sum': reward_sum
                # }
                pickle.dump(model, pickle_out)
                pickle_out.close()



            # episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
            observation = env.reset()
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            mystring = "Resetting env, episode reward was %.3f running mean: %.3f" % (reward_sum, running_reward)
            print(mystring.encode("utf-8").decode("ascii"))
            reward_sum = 0
            prev_observations = None

if __name__ == '__main__':
    main()



