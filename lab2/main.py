import numpy as np
from generator import Generator

if __name__ == "__main__":
    gen = Generator()
    x, y = gen.data_generator()
    gen.draw(x)
    print(y)
