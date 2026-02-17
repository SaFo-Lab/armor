# adapted from https://github.com/thu-ml/STAIR

from typing import Optional, Any, List
from score import evaluate_step, evaluate_answer

    
reward_type = ['strategy', 'intent', 'policy', 'answer']

class node:
    parent: Optional[Any] = None
    children: Optional[List[Any]] = None
    prompt_dict: dict = None
    trajectory: List[str] = [] # Partial solution so far
    is_terminal: bool = False
    depth: int = 0
    max_depth: int = 10
    reward: Optional[dict] = None # Reward provided by ORM for terminal node
    node_reward: Optional[int] = None # Reward provided by ORM for terminal node
    node_reward_type: Optional[str] = None 

    # Initialize
    def __init__(self, parent, output):
        self.parent = parent
        if parent != None:
            self.prompt_dict = parent.prompt_dict
            self.trajectory = parent.trajectory + [output]
            self.depth = parent.depth + 1
            self.max_depth = parent.max_depth
            if output != None:
                self.node_reward_type = reward_type[self.depth - 1]
                if self.depth <=3:
                    self.node_reward = evaluate_step(self.prompt_dict, output, self.node_reward_type)
                else:
                    self.reward = evaluate_answer(self.prompt_dict, output)
                    self.is_terminal = True
        self.children = []

    # Add child
    def add_child(self, child):
        self.children.append(child)
        
    def get_children(self):
        return  self.children


    # Return node score for required score type
    def get_node_reward(self):
        return self.node_reward


    # Return dict about node information
    def node_info(self):
        info = {}
        info["prompt_dict"] = self.prompt_dict
        info["trajectory"] = self.trajectory
        info["is_terminal"] = self.is_terminal
        info["is_valid_selected"] = self.is_valid_selected
        info["depth"] = self.depth
        info["reward"] = self.reward
        info["node_reward"] = self.node_reward

        return info

    # Set param (only needed for root node)
    def set_param(self, max_depth, question, depth, trajectory):
        self.max_depth = max_depth
        self.prompt_dict = question
        self.depth = depth
        self.trajectory = trajectory
