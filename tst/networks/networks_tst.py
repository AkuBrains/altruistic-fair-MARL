import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from networks import networks_factory, NETWORKS

if __name__=="__main__":
    print(NETWORKS)
    pass