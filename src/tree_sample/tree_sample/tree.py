# adapted from https://github.com/thu-ml/STAIR

from tree_sample.node import node
from typing import Optional
import queue

reward_type = ['strategy', 'intent', 'policy', 'answer']
class tree:
    root: Optional[node] = None
    tree_size: int = 0 

    def __init__(self, root_node):
        self.root = root_node
        self.tree_size = 1

    # Expand one node
    def add_node(self, old_node:node, child_output):
        new_node = node(old_node, child_output)
        old_node.add_child(new_node)
        self.tree_size += 1


    # Return all node information in the tree
    def show_tree(self):
        result = {}
        result["tree_size"] = self.tree_size
        tags = queue.Queue()
        nodes = queue.Queue()
        tags.put("0")
        nodes.put(self.root)
        while not tags.empty():
            tag = tags.get()
            node = nodes.get()
            result[tag] = node.node_info()
            for index in range(len(node.children)):
                tags.put(tag+"."+str(index))
                nodes.put(node.children[index])
        return result
