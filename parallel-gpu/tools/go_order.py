import json
import os

# find this file path
file_path = os.path.dirname(os.path.realpath(__file__))

class GO_order:

    def __init__(self,
                 obo_file:str=f'{file_path}/Data/go-basic.obo',# path to obo file
                 exludeGO:bool=False, # exclude GO terms
                 ):
        self.obo_file = obo_file
        self.exludeGO = exludeGO
        self.child_parent_dirct = None
        self.child_parent_full = None
        self.backpro_order_dict = {}
        self.parse_obo_file(obo_file=self.obo_file, exludeGO=self.exludeGO)

    ### Section 1: Parse the obo file and get the direct parent terms ###
    def parse_obo_file(self,obo_file:str=f'{file_path}/Data/go-basic.obo', loadCache:bool=True, exludeGO:bool=True):

        child_parent_dirct_path = f'{file_path}/Data/child_parent_dirct.json'
        child_parent_full_path = f'{file_path}/Data/child_parent_full.json'
        # load the cache
        if loadCache:
            if os.path.exists(child_parent_dirct_path):
                self.child_parent_dirct = self.json_load(child_parent_dirct_path)
                print('Load child_parent_dirct from cache')
            if os.path.exists(child_parent_full_path):
                self.child_parent_full = self.json_load(child_parent_full_path)
                print('Load child_parent_full from cache')
            if self.child_parent_dirct is not None and self.child_parent_full is not None:
                return

        # check if the file exists, if not, download it
        if not os.path.exists(obo_file):
            curr_obo_url = 'http://current.geneontology.org/ontology/go-basic.obo'
            print("File does not exist, downloading it now")
            print(f"Downloading {curr_obo_url} to {obo_file}")
            os.system(f"wget {curr_obo_url} -O {obo_file}")
            print("Download complete")
        
        # parse the dirct parents term by checking is_a
        if self.child_parent_dirct is None:
            # get the direct parent terms from the obo file
            child_parent_dirct = self.get_child_parent_from_obo(obo_file=obo_file, exludeGO=exludeGO)
            # save the dirct parent terms
            self.child_parent_dirct = child_parent_dirct
            self.json_dump(child_parent_dirct, child_parent_dirct_path)
            print('Save child_parent_dirct to cache')

        # get the complete parent terms
        if self.child_parent_full is None:
            child_parent_full = {}
            for term in child_parent_dirct:
                child_parent_full[term] = self.find_all_parents(term, child_parent_dirct)
            self.child_parent_full = child_parent_full
            self.json_dump(child_parent_full, child_parent_full_path)
            print('Save child_parent_full to cache')
    
    def get_child_parent_from_obo(self, obo_file:str=f'{file_path}/Data/go-basic.obo', exludeGO:bool=True):
        # get the direct parent terms from the obo file
        with open(obo_file, 'r') as f:
            curr_term = None
            child_parent_dirct = {}
            for line in f:
                if line.startswith("id:"):
                    curr_term = {}
                    curr_term["id"] = line.split()[1]
                elif line.startswith("is_a:"):
                    if "is_a" not in curr_term:
                        curr_term["is_a"] = set()

                    # adding the parent terms to the current term if not in the excludeGO list
                    if exludeGO:
                        if line.split()[1] not in self.excludeGO():
                            curr_term["is_a"].add(line.split()[1])
                    else:
                        curr_term["is_a"].add(line.split()[1])

                elif line.startswith("[Term]"):
                    # the "[Term]" line marks the beginning of a new term
                    # so we can add the current term to the dictionary
                    if curr_term is not None:
                        if "is_a" in curr_term:
                            child_parent_dirct[curr_term["id"]] = curr_term["is_a"]
                else:
                    continue # skip other lines     
        return child_parent_dirct      

    def json_dump(self, json_dict:dict, path:str):
        json_dict = dict([(k, list(v)) for k, v in json_dict.items()])
        json.dump(json_dict, open(path, 'w'))
    
    def json_load(self, path:str):
        json_dict = json.load(open(path, 'r'))
        json_dict = dict([(k, set(v)) for k, v in json_dict.items()])
        return json_dict
        
    def find_all_parents(self, term, child_parent_dirct):
        """
        find all parent terms of the input term recursively

        Parameters
        ----------
        term : str 
            the input term
        child_parent_dirct : set
            the set of all parent terms
        """
        if term not in child_parent_dirct:
            return set()
        else:
            parent_terms = child_parent_dirct[term] # get the dirct parent terms
            for parent_term in parent_terms:
                parent_terms = parent_terms.union(self.find_all_parents(parent_term, child_parent_dirct)) # get the parent terms of parent terms recursively
            return parent_terms # return the parent terms of the input term

    def excludeGO(self):
        excludeGO = {'GO:0005515','GO:0005488','GO:0003674','GO:0008150','GO:0005575'}
        return excludeGO

    def reverse_dict(self, child_parent_dirct:dict):
        """
        reverse the key and value of a dictionary
        args:
            child_parent_dirct: dict of child as key and parents as set of values
        return:
            parent_child_dirct: dict of parent as key and children as set of values
        """

        parent_child_dirct = {}
        for child, parents in child_parent_dirct.items():
            for parent in parents:
                if parent not in parent_child_dirct:
                    parent_child_dirct[parent] = set()
                parent_child_dirct[parent].add(child)
                if child not in parent_child_dirct:
                    parent_child_dirct[child] = set()

        return parent_child_dirct
    
    ### END of Section 1 ###

    ### Section 2: backpropagation & find toplogical order ###
    def connect_node(self, graph:dict, terms:set, parent:str, child:str,\
                     child_set:set, pairs:set, processed:set):
        """
        The go terms we not interested mighe be in the middle of the graph
        so we need to connect the parent and child terms together to skipped
        the middle terms.
        pairs is used to avoid duplicate processing of the same pairs
        processed is used to avoid duplicate processing of the same child term
        for current parent terms
        example:
        input:
            graph = {'GO:0000001': {'GO:0000002', 'GO:0000003'},
                        'GO:0000002': {'GO:0000004', 'GO:0000005'}}
            parent = 'GO:0000001'
            child = 'GO:0000002'
            processed = set(); pairs = set()
        output:
            {'GO:0000002', 'GO:0000003', 'GO:0000004', 'GO:0000005'},

        args:
            graph: dict of parent as key and children as set of values
            terms: set of all terms we interested in
            parent: str of parent term dict keys
            child: str of child term dict keys
            child_set: set of child terms that connected to parent
            pairs: set of (parent, child) pairs processed
            processed: set of terms processed for current parent
        return:
            child_set: set of child terms that connected to parent
            pairs: set of (parent, child) pairs processed
            processed: set of terms processed for current parent
        """

        # connect grandchild to parent
        child_set = child_set.union(graph[child])
        # find child not in terms that need to be processed and skipped
        child_not_in_terms = child_set.difference(terms)
        # avoid duplicate processing of the child terms
        child_not_in_terms = child_not_in_terms.difference(processed)
        for cur_child in child_not_in_terms:
            # avoid duplicate processing of the same pairs
            if (parent, cur_child) in pairs:
                continue
            processed.add(cur_child)
            pairs.add((parent, cur_child))
            child_set, pairs, processed = self.connect_node(graph, terms, parent, cur_child,\
                                          child_set, pairs, processed)
        return child_set, pairs, processed

    def graph_fit_term(self, graph:dict, root:str, terms:set):
        # graph: dict of parent as key and dirct children as set of values
        
        # first get a subset of the graph that derive from the root
        subset_graph = self.get_subgraph(graph, root)
        # then remove the terms we not interested and connect the its parent and child
        pairs = set()
        for parent, childs in subset_graph.items():
            childs_copy = childs.copy()
            for child in childs_copy:
                if child not in terms:
                    proccessed = {child}
                    child_set = subset_graph[parent].copy() # the child set could be changed in the connect_node function
                    child_set, pairs, proccessed = self.connect_node(subset_graph, terms, parent, child,\
                                                                      child_set, pairs, proccessed)
                    subset_graph[parent] = child_set
                    pairs.add((parent, child))
        # remove terms not in terms of interest
        subset_graph = self.remove_not_in_terms(subset_graph, root, terms)
        # remove duplicate edges
        subset_graph = self.remove_duplicate_edges(subset_graph)
        return subset_graph

    def get_subgraph(self, graph:dict, root:str):
        subgraph = {}
        stack = [root]
        while len(stack) > 0:
            node = stack.pop()
            subgraph[node] = graph[node]
            stack.extend(graph[node])
        return subgraph

    def remove_not_in_terms(self, graph:dict,root:str,terms:set):
        """
        Remove terms not in terms from graph
        For terms not in graph, add them to graph as a leaf node
        connected it to the root

        Args:
            graph (dict): dict of parent as key and children as set of values
            root (str): root term
            terms (set): set of terms we interested in

        Returns:
            graph (dict): dict of parent as key and children as set of values
                that only contains terms we interested in
        """
        new_graph = {}
        for parent, childs in graph.items():
            if parent in terms or parent == root:
                new_graph[parent] = set()
                for child in childs:
                    if child in terms:
                        new_graph[parent].add(child)
        for term in terms:
            if term not in new_graph:
                new_graph[term] = set()
                new_graph[root].add(term)
        return new_graph
    
    def remove_duplicate_edges(self, graph:dict):
        """
        For each parent, remove its children that are also children of its children
        This is to avoid duplicates in the backpropagation

        Args:
            graph (dict): dict of parent as key and children as set of values
        
        Returns:
            graph (dict): dict of parent as key and children as set of values
                that does not contain duplicates
        """
        graph = graph.copy()
        for parent in graph:
            childs = graph[parent].copy()
            to_remove = set()
            for child in childs:
                child_v = graph[child]
                intersection = childs.intersection(child_v)
                to_remove = to_remove.union(intersection)
            graph[parent] = childs.difference(to_remove)
        return graph

    def bfs(self, graph:dict, root:str):
        """
        Breadth-first search of the graph
        args:
            graph: dict of parent as key and children as set of values
            root: str of root term dict keys
        return:
            bfs_order: list of terms in bfs order (child, parent) that
                could used for backpropagation
        """
        quene = [(root, root)]
        bfs_order = []
        while len(quene) > 0:
            child, parent = quene.pop(0)
            bfs_order.append((child, parent))
            for grandchild in graph[child]:
                quene.append((grandchild, child))
        bfs_order = [ i for i in bfs_order if i[1] != root]
        return bfs_order[::-1]
    
    def backpro_order(self, graph:dict, root:str, terms:set):
        """
        Get the order of backpropagation
        """
        subset_graph = self.graph_fit_term(graph, root, terms)
        bfs_order = self.bfs(subset_graph, root)
        return bfs_order

    ### End of Section 2: Graph Processing ###

    ### Section 3: initialize the order of backpropagation ###
    def init_order(self):
        """
        Initialize the order of backpropagation
        """
        aspects = ['BP', 'MF', 'CC']
        for aspect in aspects:
            self.backpro_order_dict[aspect] = self.init_one_aspect(aspect)

    def init_one_aspect(self, aspect:str):
        """
        Initialize the order of backpropagation for one aspect
        """
        aspect = aspect.upper()
        go_obj = GO(aspect)
        root = go_obj.root
        terms = set(go_obj.go_list)
        parent_child_dirct = self.reverse_dict(self.child_parent_dirct)
        bfs_order = self.backpro_order(parent_child_dirct, root, terms)
        bfs_order = self.order2num(go_obj.go2num, bfs_order)
        # convert bfs_order to 
        return bfs_order
    
    def get_order(self, aspect:str):
        """
        Get the order of backpropagation for one aspect
        """
        if aspect not in self.backpro_order_dict:
            self.init_order()
        return self.backpro_order_dict[aspect]
    
    def order2num(self,go2num:str, order:list):
        """
        Convert order of backpropagation to number
        """
        num_order = []
        for go, parent in order:
            num_order.append((go2num[go], go2num[parent]))
        return num_order

    def brute_order(self, aspect:str):
        """
        Get the order of backpropagation for one aspect
        """
        aspect = aspect.upper()
        go_obj = GO(aspect)
        root = go_obj.root
        terms = set(go_obj.go_list)

        parent_child_full = self.reverse_dict(self.child_parent_full)

        leaf_nodes = terms.difference(parent_child_full.keys())
        for parent, childs in parent_child_full.items():
            childs = set(childs) & terms
            if len(childs) == 0:
                leaf_nodes.add(parent)


        bfs_order = []
        for leaf in leaf_nodes:
            if leaf in terms:
                parent = self.child_parent_full[leaf].copy()
                for p in parent:
                    if p in terms:
                        bfs_order.append((go_obj.go2num[leaf], go_obj.go2num[p]))

        return bfs_order


   


class GO:

    def __init__(self, aspect:str, # one of the following: 'BP', 'MF', 'CC'
                 term_path:str=None # path to all go terms
                 ): 
        self.aspect = self.check_aspect(aspect)
        self.term_path = term_path if term_path else self.get_term_path()
        self.go_list = self.read_go()
        self.go2num = {go: i for i, go in enumerate(self.go_list)}
        self.num2go = {i: go for i, go in enumerate(self.go_list)}

    @property
    def root(self):
        root_dict = {
            'BP': 'GO:0008150',
            'MF': 'GO:0003674',
            'CC': 'GO:0005575'
        }
        return root_dict[self.aspect]
    
    def get_term_path(self):
        term_path = f'{file_path}/Data/{self.aspect.lower()}_term_list'
        return term_path

    def read_go(self, sort:bool=False):
        term_list = set()
        with open(self.term_path, 'r') as f:
            for line in f:
                go_term = line.strip()
                term_list.add(go_term)
        term_list = list(term_list)
        if sort:
            term_list.sort()
        return term_list

    def go2vec(self, go_list:list):
        return [self.go2num[go] for go in go_list]
    
    def vec2go(self, go_list:list):
        return [self.num2go[go] for go in go_list]
    
    def check_aspect(self, aspect:str):
        aspect = aspect.upper()
        assert aspect in ['BP', 'MF', 'CC'], 'aspect must be one of the following: BP, MF, CC'
        return aspect



if __name__ == "__main__":
    go_order = GO_order()