import datetime
import numpy as np


def create_long_index():
	"""
	Create random long int combining the time of creation and a a random number
	:return: a long int index
	"""
	
	first_part  = datetime.datetime.now().strftime("%s%f") #Create a str with second+microsecond
	second_part = str(np.random.randint(100,999))
	joined_part = first_part+second_part

	return int(joined_part)

	