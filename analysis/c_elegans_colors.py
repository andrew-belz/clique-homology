import csv

def c_elegans_colors():
    
    with open("c_elegans_data\maleChemicalFunctionGroupColoring.csv", "r") as file:
        reader = csv.reader(file)

    neurons = []
    group = []
    for row in reader:
        neurons.append(row[0])
        group.append(row[1])

    