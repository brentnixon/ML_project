l#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    import numpy as np

    for i in range(len(ages)):
        age = int(ages[i])
        nw = float(net_worths[i])
        err = abs(float(predictions[i] - net_worths[i]))
        tup = (age, nw, err)
        cleaned_data.append(tup)

    ext = int(.9*len(cleaned_data))
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2]) 
    cleaned_data = cleaned_data[ : ext]

    return cleaned_data

