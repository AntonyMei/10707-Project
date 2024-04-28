import torch
from onpolicy.algorithms.quartz_ppo_dual.algorithm.quartz_ppo_network import QuartzPPONetwork
from onpolicy.utils.quartz_utils import update_linear_schedule


class QuartzPPOModel:
    def __init__(self,
                 # initial register embedding
                 reg_degree_types,                  # number of different register degrees
                 reg_degree_embedding_dim,          # dimension of register feature (degree) embedding
                 # initial qubit embedding
                 gate_is_input_embedding_dim,       # dimension of gate feature (is_input) embedding
                 # graph convolution layers
                 num_gnn_layers,                    # number of graph neural network layers
                 reg_representation_dim,            # dimension of register representation
                 gate_representation_dim,           # dimension of gate representation
                 # optimization process
                 device,                            # device the model is on
                 rank,                              # rank of current model (ddp)
                 allow_nop,                         # allow nop (0, 0) in action space
                                                    # (Initial: True, Physical: False)
                 lr,                                # learning rate
                 opti_eps,                          # eps used in adam
                 weight_decay,                      # weight decay
                 ):
        # network
        self.device = device
        self.rank = rank
        self.actor_critic = QuartzPPONetwork(reg_degree_types=reg_degree_types,
                                             reg_degree_embedding_dim=reg_degree_embedding_dim,
                                             gate_is_input_embedding_dim=gate_is_input_embedding_dim,
                                             num_gnn_layers=num_gnn_layers,
                                             reg_representation_dim=reg_representation_dim,
                                             gate_representation_dim=gate_representation_dim,
                                             device=device,
                                             rank=rank,
                                             allow_nop=allow_nop)
        self.actor_critic.to(device)

        # optimizer
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(),
                                          lr=lr, eps=opti_eps, weight_decay=weight_decay)

    def lr_decay(self, cur_episode, total_episodes):
        update_linear_schedule(self.optimizer, cur_episode, total_episodes, self.lr)

    def get_actions(self,
                    circuit_batch,                      # a list of GraphState objects
                    device_edge_list_batch,             # a list of device_edge_list (each has shape=(#edges, 2))
                    physical2logical_mapping_batch,     # a list of physical -> logical mapping (each is a np.array)
                    logical2physical_mapping_batch,     # a list of logical -> physical mapping (each is a np.array)
                    action_space_batch,                 # a list of action spaces (each is a list of (x, y))
                    is_initial_phase_batch              # a list of bools, # = batch size
                    ):
        """
        Compute actions and value function predictions for the given inputs.
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: value, selected action, log_probability of selected action (each is a list)
        """
        selected_action_list, selected_action_prob_list, value_batch = \
            self.actor_critic.policy_value_forward(circuit_batch=circuit_batch,
                                                   device_edge_list_batch=device_edge_list_batch,
                                                   physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                   logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                   action_space_batch=action_space_batch,
                                                   is_initial_phase_batch=is_initial_phase_batch)
        return value_batch, selected_action_list, selected_action_prob_list

    def get_values(self,
                   circuit_batch,                   # a list of GraphState objects
                   device_edge_list_batch,          # a list of device_edge_list (each has shape=(#edges, 2))
                   physical2logical_mapping_batch,  # a list of physical -> logical mapping (each is a np.array)
                   logical2physical_mapping_batch,  # a list of logical -> physical mapping (each is a np.array)
                   is_initial_phase_batch,          # a list of bools, # = batch size
                   ):
        """
        Get value function predictions.
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
        output: a batch of values
        """
        value_batch = self.actor_critic.value_forward(circuit_batch=circuit_batch,
                                                      device_edge_list_batch=device_edge_list_batch,
                                                      physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                      logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                      is_initial_phase_batch=is_initial_phase_batch)
        return value_batch

    def evaluate_actions(self,
                         circuit_batch,                     # a list of GraphState objects
                         device_edge_list_batch,            # a list of device_edge_list (each has shape=(#edges, 2))
                         physical2logical_mapping_batch,    # a list of physical -> logical mapping (each is a np.array)
                         logical2physical_mapping_batch,    # a list of logical -> physical mapping (each is a np.array)
                         action_space_batch,                # a list of action spaces (each is a list of (x, y))
                         action_id_batch,                   # a list of selected action ids
                         is_initial_phase_batch             # a list of bools, # = batch size
                         ):
        """
        Get action log_prob / entropy and value function predictions for actor update.
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
                action_id_batch: list of selected actions index
        output: selected_action_log_probs: [#batchsize, 1]
                dist_entropy: a number (as a tensor)
                value_batch: [#batchsize, 1]
        """
        selected_action_log_probs, dist_entropy, value_batch = \
            self.actor_critic.evaluate_action_value_forward(circuit_batch=circuit_batch,
                                                            device_edge_list_batch=device_edge_list_batch,
                                                            physical2logical_mapping_batch=physical2logical_mapping_batch,
                                                            logical2physical_mapping_batch=logical2physical_mapping_batch,
                                                            action_space_batch=action_space_batch,
                                                            action_id_batch=action_id_batch,
                                                            is_initial_phase_batch=is_initial_phase_batch)
        return value_batch, selected_action_log_probs, dist_entropy

    # def evaluate_actions_original(self,
    #                               circuit_batch,                     # a list of GraphState objects
    #                               device_edge_list_batch,            # a list of device_edge_list (each is a list of edges)
    #                               physical2logical_mapping_batch,    # a list of physical -> logical mapping (each is a map)
    #                               logical2physical_mapping_batch,    # a list of logical -> physical mapping (each is a map)
    #                               action_space_batch,                # a list of action spaces (each is a list of (x, y))
    #                               action_id_batch,                   # a list of selected action ids
    #                               is_initial_phase_batch             # a list of bools, # = batch size
    #                               ):
    #     """
    #     Get action log_prob / entropy and value function predictions for actor update.
    #     input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
    #             logical2physical_mapping_batch: list of observations
    #             action_space_batch: list of decoded action space (see utils.DecodePyActionList)
    #             action_id_batch: list of selected actions index
    #     output: value, log probabilities of the input action, distribution entropy (each is a list)
    #     """
    #     action_log_prob_batch, dist_entropy_batch, value_batch = \
    #         self.actor_critic.evaluate_action_value_forward_original(circuit_batch=circuit_batch,
    #                                                                  device_edge_list_batch=device_edge_list_batch,
    #                                                                  physical2logical_mapping_batch=physical2logical_mapping_batch,
    #                                                                  logical2physical_mapping_batch=logical2physical_mapping_batch,
    #                                                                  action_space_batch=action_space_batch,
    #                                                                  action_id_batch=action_id_batch,
    #                                                                  is_initial_phase_batch=is_initial_phase_batch)
    #     return value_batch, action_log_prob_batch, dist_entropy_batch

    def act(self,
            circuit_batch,                      # a list of GraphState objects
            device_edge_list_batch,             # a list of device_edge_list (each has shape=(#edges, 2))
            physical2logical_mapping_batch,     # a list of physical -> logical mapping (each is a np.array)
            logical2physical_mapping_batch,     # a list of logical -> physical mapping (each is a np.array)
            action_space_batch,                 # a list of action spaces (each is a list of (x, y))
            deterministic=False):
        """
        Compute actions using the given inputs.
        input:  circuit_batch, device_edge_list_batch, physical2logical_mapping_batch,
                logical2physical_mapping_batch: list of observations
                action_space_batch: list of decoded action space (see utils.DecodePyActionList)
        output: a list of selected actions
        """
        selected_action_list, _ = \
            self.actor_critic.policy_forward(circuit_batch=circuit_batch,
                                             device_edge_list_batch=device_edge_list_batch,
                                             physical2logical_mapping_batch=physical2logical_mapping_batch,
                                             logical2physical_mapping_batch=logical2physical_mapping_batch,
                                             action_space_batch=action_space_batch,
                                             deterministic=deterministic)
        return selected_action_list
