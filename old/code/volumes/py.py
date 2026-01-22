import os
import multiprocessing
print("Logical CPU cores:", os.cpu_count())
print("Physical CPU cores:", multiprocessing.cpu_count())
