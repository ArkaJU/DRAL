from agent import Agent
from data_loader import Market1501, load_model
from feature_extractor import model
from distances import calculate_similarity

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
from torchvision import transforms

if T.cuda.is_available():                        #change default type else errors pop up, find a better way
  T.set_default_tensor_type(T.cuda.FloatTensor)

Ns = 30
input_dims = (Ns+1)*(Ns+1)
fc1_dims = 256
fc2_dims = 256
fc3_dims = 256

agent = Agent(ALPHA=0.001, input_dims=input_dims, fc1_dims=fc1_dims, fc2_dims=fc2_dims, fc3_dims=fc3_dims, n_actions=Ns)
#model = Model()                                      #might be renamed to the particular architecture
#tau_r = MakeDataset()
tau_p = []
GAMMA = 0.9
cost = 0
max_epoch = 30
Kmax = 10 

feature_extractor = model()
market = Market1501()

print("==> Start training")
for epoch in range(max_epoch):

    expected_return = 0
    action_buffer = []

    query, query_label, g, g_labels = market.nextbatch(n=30)

    # print(g.shape)                #torch.Size([30, 3, 224, 224])
    # print(g_labels.shape)           #torch.Size([30])
    # print(query.shape)            #torch.Size([1, 3, 224, 224])
    # print(query_label.shape)      #torch.Size([1])
              
    g_features = feature_extractor(g)                         #have to take care how the image features are concat->row-wise
    query_feature = feature_extractor(query) 
    state = calculate_similarity(T.cat((query_feature, g_features)))
    
    #print(f"g_features.shape : {g_features.shape}")           #torch.Size([30, 2048])
    #print(f"query_feature.shape : {query_feature.shape}")     #torch.Size([1, 2048])
    #print(f"state:{state.shape}")                             #torch.Size([31, 31])
    
    for t in range(Kmax):
      
      logits = agent(state.flatten())
      #logits_ = logits.detach().clone()                 #might be buggy
      #print(logits)                                                  
      action = agent.take_unique_action(logits, action_buffer)            #keep track of the actions already taken, can't take those actions later  
      action_buffer.append(action)
      #print(action_buffer) 
      gk_feature = g_features[action]
      g_label = g_labels[action]
      #print(g_label)
      #print(gk_feature.shape)                           #torch.Size([2048])
      agent.sim_mat.find_max_min_distances_in_batch(query_feature, gk_feature)           #useful for reward calc
      
      #label = ActiveLearning(query, gk)                  #dummy function #should take in images not feature
      # print(g_label)
      # print(query_label)
      # print("**************")
      label = 1 if g_label == query_label  else -1               #PROBLEM OF CLASS IMBALANCE, ALMOST ALL ARE FALSE
      state = agent.update_state(state, label, action+1)  #action+1 since max(action_buffer)=29 but state has first row(col) of query
      #print(f"label : {label}")
      agent.sim_mat.state_matrix = state
      tau_p.append((query_feature, gk_feature, label))                   #change this later, it takes images
      reward = agent.compute_reward(label)
      expected_return -= GAMMA * reward 
    #print(action_buffer)
    agent.optimizer.zero_grad()
    expected_return.backward()
    agent.optimizer.step()
    #model = ModelTrain()
    #print(f"Outside inner loop1: {state.shape}")
    agent.sim_mat.reset_distances()
    print(f"epoch {epoch+1}/{max_epoch}\t Expected return: {expected_return.item()}\t")
