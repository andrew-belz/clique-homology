import json


def c_elegans_colors() -> tuple[list[str], list[int], list[str]]:
    """Return the neurons and their functional groupings. 

    Returns:
        tuple[list[str], list[int], list[str]]: list containing a 
            neuroscientific name for each neuron, list containing an index for
            each neuron, list containing a color for each neuron. Organized by
            list index.
    """

    # see wrangling.py for details about how c_elegans_data\neuron_indices.json
    #   is produced.
    with open(r"c_elegans_data\neuron_indices.json", "r") as file:
        data = json.load(file)

        neurons = [] # name for each neuron (node)
        I = [] # associated index for each neuron (node)
        groups = [] # color label for each neuron (node)

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

    