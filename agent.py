import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class SimilarityMatrix:

  def __init__(self, N):
    self.state_matrix = T.zeros((N+1, N+1))
    #self.kreciprocal_matrix = T.zeros((N+1, N+1))
    self.max_positive_distance = 0
    self.min_negative_distance = 1/T.zeros((1,))                 #potential error source  
  
  #def k_reciprocal(self, k=15):
  
  def update_distances(self, query_feature, gk_feature):
    #print(self.min_negative_distance)
    #print(self.max_positive_distance)
    dist = mahalanobis_dist_from_vectors(query_feature, gk_feature.reshape(1,-1))
    #print(dist)
    #print("****************************************")
    self.max_positive_distance = max(self.max_positive_distance, dist).reshape(1)
    self.min_negative_distance = min(self.min_negative_distance, dist).reshape(1)

  def reset_distances(self):
    self.max_positive_distance = 0
    self.min_negative_distance = 1/T.zeros((1,))                      #potential error source



class Agent(nn.Module):   #architecture doubt: Ns=30->flattened state matrix:961, 3 fc layers with 256 in paper

  def __init__(self, ALPHA, input_dims, fc1_dims, fc2_dims, fc3_dims, n_actions):
      super(Agent, self).__init__()
      self.input_dims = input_dims
      self.fc1_dims = fc1_dims
      self.fc2_dims = fc2_dims
      self.fc3_dims = fc3_dims
      self.n_actions = n_actions

      self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
      self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
      self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
      self.output = nn.Linear(self.fc3_dims, self.n_actions)
      self.optimizer = optim.Adam(self.parameters(), lr=ALPHA)

      self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
      self.to(self.device)

      self.sim_mat = SimilarityMatrix(Ns)

  def forward(self, state):
      state = T.Tensor(state).to(self.device)
      x = F.relu(self.fc1(state))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = F.relu(self.output(x))                            #add activation
      return x
  
  def compute_reward(self, label, margin=0.2):
    #print(self.sim_mat.max_positive_distance.shape)
    #print(self.sim_mat.min_negative_distance.shape)
    reward = margin + label*(self.sim_mat.max_positive_distance - self.sim_mat.min_negative_distance)
    #print(f"Reward:{reward}")
    return reward

  def update_state(self, state, label, g_k, threshold=0.4):
    #print(label)
    if label == 1:
      z = (state[:, 0] + state[:, g_k]) / 2
      state[:, 0] = z
      state[0, :] = z
      state[0, g_k] = 1
      state[g_k, 0] = 1

    else:
      z = state[:, g_k].detach().clone()                     #might be buggy
      z[z<threshold] = 0
      state[:, 0] = T.clamp(state[:, 0] - z, min=0)
      state[0, g_k] = 0
      state[g_k, 0] = 0

    z = state[:, 0]
    state[0, :] = z
    state.fill_diagonal_(0)                                     #look into this
    #print(state)
    #print("**************************************")
    #assert(np.diagonal(state).any() == False)  #sanity check, diagonal elements should be zero
    return state
  
  def take_unique_action(self, logits, action_buffer):
    max_so_far = [-1/T.zeros((1,)), 0]
    for i in range(logits.shape[0]):
      if i not in action_buffer and logits[i] > max_so_far[0]:
        max_so_far[0], max_so_far[1] = logits[i], i
    return max_so_far[1]
