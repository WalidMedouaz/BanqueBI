import pandas as pd
import discretisation as disc
import normalisation as norm
import numerisation as num

def main():
	num.numerisation(norm.normalisation(disc.discretisation()))

main()
