action = ['delete_node', 'delete_edge', 'filter', 'find_path']

delete_node_action = ['0', '1', '2', '3', '4', '5']

delete_edge_action = [('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '3'), ('4', '1'), ('5', '2')]

# filter action means filter path from the graph
filter_action = [['0', '1', '2', '3'], ['5', '0', '1']]

find_path_condition = ['Delay < 20', "HopCount < 5", "Delay < 15"]

find_path_source = ['0', '1', '2', '3', '4', '5']

find_path_target = ['0', '1', '2', '3', '4', '5']
