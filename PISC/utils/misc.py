import numpy as np

def pairwise_swap(a,l):
	tempodd = a[...,1:l:2].copy()
	tempeven = a[...,2:l:2].copy()
	temp = a.copy()
	temp[...,1:l:2] = tempeven
	temp[...,2:l:2] = tempodd
	return temp[...,:l]

