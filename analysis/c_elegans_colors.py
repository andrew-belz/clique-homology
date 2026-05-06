import json

def c_elegans_colors():
    """
    Return the neurons and their functional groupings. 
    """
    
    with open(r"c_elegans_data\neuron_indices.json", "r") as file:
        data = json.load(file)

        neurons = [] # neuron names
        I = [] # associated indices
        groups = [] # neuron node labels

        for neuron in data:

            # make sure the neuron has a valid group assigned
            try:
                i = data[neuron][0]
                group = data[neuron][1]

                neurons.append(neuron)
                I.append(i)
                groups.append(group)

            except:
                print(f"{neuron} has no assigned group.")

        return neurons, I, groups

        
if __name__ == "__main__":
    print(c_elegans_colors())

    