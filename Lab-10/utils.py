# Defines the implementation of the FedAvg aggregation algorithm

def fed_avg(weights_list):
    avg_weights = {}
    for key in weights_list[0].keys():
        avg_weights[key] = sum([w[key] for w in weights_list]) / len(weights_list)
    return avg_weights