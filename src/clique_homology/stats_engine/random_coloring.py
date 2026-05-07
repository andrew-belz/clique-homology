from random import choice

def random_coloring(
    colors: list[str],
    proportional: bool = False
) -> list[str]:
    """Generates a random coloring based on an existing coloring.

    Args:
        colors (list[Color]): A coloring to serve as the basis for the new
            random coloring.
        proportional (bool, optional): True means the colors for the random
            coloring will be chosen according to the proportion of colors in the
            existing coloring. False means each color has an equal probability
            of being chosen. Defaults to False.

    Raises:
        TypeError: raised if some element of the list of colors isn't a string.
        
    Returns:
        list[Color]: A random coloring
    """
    # Validate that all the colors in the colors list are strings
    if not all(isinstance(color, str) for color in colors):
        raise TypeError("All input color values must be strings.")

    # Initialize new coloring and a palette to serve as the source for colors
    new_coloring: list[str] = []
    palette: list[str]

    # If proportional == True, make the palette just a copy of colors. If not,
    #   make the palette a list of all the colors in colors, without duplicates.
    palette = list(colors) if proportional else list(set(colors))
    
    # Append a random color for each node in the original coloring, from the
    #   palette
    for _ in range(len(colors)):
        new_coloring.append(choice(palette))
            
    return new_coloring