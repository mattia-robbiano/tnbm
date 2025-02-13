import math
from dataset import get_bars_and_stripes
from  born_machine_1 import QCBM, MMD
import json


# Importing configuration file
with open("parameters.json", "r") as f:
    config = json.load(f)
SAMPLE_BITSTRING_DIMENSION = config['SAMPLE_BITSTRING_DIMENSION']


# Output configuration parameters to user
print()
print("Configuration:")
print(f"bitsting samples dimension: {SAMPLE_BITSTRING_DIMENSION}")
if math.sqrt(SAMPLE_BITSTRING_DIMENSION).is_integer() == False:
    raise ValueError("bitstring samples dimension must be a perfect square!")
print(f"number of different samples:{2**(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))*2-2}")
print()


def main():
    dataset = get_bars_and_stripes(int(math.sqrt(SAMPLE_BITSTRING_DIMENSION)))
    print(dataset.shape)
    
if __name__ == "__main__":
    main()
