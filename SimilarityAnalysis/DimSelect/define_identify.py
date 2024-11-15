import numpy as np
import pandas as pd
from scipy import optimize
import argparse
import os
import cPickle as pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from embedding_dist import cal_embedding_distance

#temp
import networkx as nx

def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--uri', type=str, default='bolt://localhost:7687',
                        help='URI for connecting to Neo4j database.')
    parser.add_argument('--user', type=str, default='neo4j',
                        help='Username for Neo4j database.')
    parser.add_argument('--password', type=str, default='password',
                        help='Password for Neo4j database.')
    parser.add_argument('--query', type=str, default='MATCH (n)-[r]->(m) RETURN n.id AS node1, m.id AS node2, r.weight AS weight',
                        help='Cypher query to retrieve graph data from Neo4j.')
    parser.add_argument('--directed', action='store_true', default=False,
                        help='Graph is directed.')
    parser.add_argument('--weighted', action='store_true', default=False,
                        help='Graph is weighted.')
    parser.add_argument('--p', type=float, default=1.0,
                        help='Return hyperparameter.')
    parser.add_argument('--q', type=float, default=1.0,
                        help='Inout hyperparameter.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source.')
    parser.add_argument('--length', type=int, default=80,
                        help='Length of walk per source.')
    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization.')
    parser.add_argument('--iter', type=int, default=1,
                        help='Number of epochs in SGD.')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers.')
    parser.add_argument('--start-dim', type=int, default=32,
                        help='Starting dimension for embedding.')
    parser.add_argument('--end-dim', type=int, default=512,
                        help='Ending dimension for embedding.')
    parser.add_argument('--step', type=int, default=10,
                        help='Step size for embedding dimension.')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    norm_loss = cal_embedding_distance(args)