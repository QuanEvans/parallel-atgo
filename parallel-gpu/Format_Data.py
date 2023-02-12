
import os
import sys
import math
from decimal import Decimal

def mean(value_list):   # mean value

    sum = 0
    for value in value_list:
        sum = sum+value

    return sum/len(value_list)

def l2(value_list):  # variance value

    mean_value = mean(value_list)

    sum = 0
    for value in value_list:
        sum = sum + (value-mean_value)*(value-mean_value)

    return math.sqrt(sum)

def format(value_list):  # l2 normalization

    mean_value = mean(value_list)
    l2_value = l2(value_list)

    for i in range(len(value_list)):

        value_list[i] = round((value_list[i]-mean_value)/l2_value, 4)

    return value_list

