import sys 
sys.path.append("..")
import os
print(os.getcwd())

from module1.cal import add

print(add(1,2))