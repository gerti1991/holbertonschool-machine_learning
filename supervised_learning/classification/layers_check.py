def check(layers):
    if not isinstance(layers, list) or len(layers) == 0:
        raise TypeError("layers must be a list of positive integers")
    for node in layers:
        if node <= 0:
            raise TypeError("layers must be a list of positive integers")