import os
import functools

import pandas as pd
import numpy as np

import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.bandits.environments import classification_environment as class_env
from tf_agents.bandits.environments import environment_utilities as env_util
from tf_agents.networks import q_network as q_net
from tf_agents.bandits.agents import linear_thompson_sampling_agent as lin_ts_agent
from tf_agents.bandits.agents import neural_epsilon_greedy_agent as eps_greedy_agent
from tf_agents.bandits.metrics import tf_metrics
import tensorflow_probability as tfp 
from tf_agents.bandits.agents.examples.v2 import trainer as tr
from absl import app

tfd = tfp.distributions

def training(unused_argv):
  
  # * Get Data
  DATASETS = 30000
  np.random.seed(42)
  data = np.random.rand(DATASETS, 66)
  df = pd.DataFrame(data, columns=[f"col_{i}" for i in range(66)])
  df_TRAINING_DATA = df.round(5)
  
  data = np.random.randint(low=1, high=201, size=DATASETS)
  df_Training_LABEL = pd.DataFrame(data, columns=['column_name'])
  
  # * Parameter
  MODEL_NAME = "TestModel"
  AGENT = "Greedy" # Can also be "LinTS"
  BATCH_SIZE = 10
  LAYERS = (300, 200, 100, 100, 50, 50)
  EPSILON = 0.01
  LR = 0.002
  AGENT_ALPHA = 10
  ROOT_DIR = 'C://' # ! Add your Root_dir 
  TRAINING_LOOPS = 20000
  STEPS_PER_LOOP = 2
  
  with tf.device('/CPU:0'):
    
    # * Training Dataset
    CONTEXT_TENSOR = tf.cast(df_TRAINING_DATA, tf.float32)
    LABEL_TENSOR = tf.cast(df_Training_LABEL, tf.int32)
    TRAINING_DATASET =  tf.data.Dataset.from_tensor_slices((CONTEXT_TENSOR, LABEL_TENSOR))
    
    # * Reward Distribution
    reward_distribution = tfd.Independent(
      tfd.Deterministic(tf.eye(200)), 
      reinterpreted_batch_ndims=2) 
    
    # * Environment
    environment = class_env.ClassificationBanditEnvironment(
      dataset = TRAINING_DATASET, 
      reward_distribution = reward_distribution, 
      batch_size = BATCH_SIZE, 
      prefetch_size=len(TRAINING_DATASET))
    
    # * Network/Agent
    match AGENT: 
      case 'Greedy':
        network = q_net.QNetwork(
          input_tensor_spec=environment.time_step_spec().observation,
          action_spec=environment.action_spec(),
          fc_layer_params=LAYERS)
        agent = eps_greedy_agent.NeuralEpsilonGreedyAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          reward_network=network,
          optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=LR),
          epsilon=EPSILON)
      case 'LinTS':
        agent = lin_ts_agent.LinearThompsonSamplingAgent(
          time_step_spec=environment.time_step_spec(),
          action_spec=environment.action_spec(),
          alpha=AGENT_ALPHA,
          dtype=tf.float32)
    
    # * Metric (Trainer)   
    optimal_reward_fn = functools.partial(env_util.compute_optimal_reward_with_classification_environment,environment=environment) 
    regret_metric = tf_metrics.RegretMetric(optimal_reward_fn)
    optimal_action_fn = functools.partial(env_util.compute_optimal_action_with_classification_environment,environment=environment) 
    suboptimal_arms_metric = tf_metrics.SuboptimalArmsMetric(optimal_action_fn)
    
    # * Trainer
    tr.train(
      root_dir=os.path.join(ROOT_DIR,MODEL_NAME),
      agent=agent,
      environment=environment,
      training_loops=TRAINING_LOOPS,
      steps_per_loop=STEPS_PER_LOOP,
      additional_metrics=[regret_metric, suboptimal_arms_metric])
    
    print('Done')

if __name__ == '__main__':
  app.run(training)
