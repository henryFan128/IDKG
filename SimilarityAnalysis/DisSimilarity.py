from neo4j import GraphDatabase
import matplotlib.pyplot as plt
import itertools
import numpy as np
from node2vec import Node2Vec
import networkx as nx
from scipy.spatial.distance import  cosine
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler

uri = "bolt://localhost:7687"  
username = "neo4j"              
password = "xxxxxxxxxxx"     
driver = GraphDatabase.driver(uri, auth=(username, password))

def get_all_diseases(tx):
    query = """
    MATCH (d:Disease)
    RETURN d.Disease_name AS disease_name
    """
    result = tx.run(query)
    return [record["disease_name"] for record in result]

def get_drug_set(tx, disease_name):
    query = """
    MATCH (d:Disease {Disease_name: $disease_name})-[]-(drug:Drug)
    RETURN drug.Drug_name AS drug_name
    """
    result = tx.run(query, disease_name=disease_name)
    return set(record["drug_name"] for record in result)

def get_graph(tx):
    query = """
    MATCH (n:Drug)-[r]->(m:Disease)
    RETURN n.Drug_name AS source, m.Disease_name AS target
    """
    result = tx.run(query)
    edges = [(record["source"], record["target"]) for record in result if record["source"] and record["target"]]
    return edges

def calculate_similarity(disease_list, model, distance_func, inv_cov_matrix=None):
    shared_similarities = []
    not_shared_similarities = []
    
    for disease_a, disease_b in itertools.combinations(disease_list, 2):
        Sa = session.execute_read(get_drug_set, disease_a)
        Sb = session.execute_read(get_drug_set, disease_b)
        
        if not Sa or not Sb:
            continue
        
        shared = bool(Sa & Sb)
        
        vec_a = model.wv[disease_a]
        vec_b = model.wv[disease_b]
        
        if distance_func == mahalanobis:
            distance = distance_func(vec_a, vec_b, inv_cov_matrix)
        else:
            distance = distance_func(vec_a, vec_b)
        
        similarity = np.exp(-distance)
        
        if shared:
            shared_similarities.append(similarity)
        else:
            not_shared_similarities.append(similarity)
    
    return shared_similarities, not_shared_similarities

def plot_similarity(shared_similarities, not_shared_similarities, title, filename):
    plt.figure(figsize=(6,4))

    kde_shared = gaussian_kde(shared_similarities)
    kde_not_shared = gaussian_kde(not_shared_similarities)

    x = np.linspace(0, 1, 100)
    plt.plot(x, kde_shared(x), label='share')
    plt.plot(x, kde_not_shared(x), label='non-share')
    
    plt.xlabel('disease similarity')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(filename, dpi=300)
    plt.show()

with driver.session() as session:
    disease_list = session.execute_read(get_all_diseases)
    edges = session.execute_read(get_graph)
    
    G = nx.Graph()
    G.add_edges_from(edges)
    
    for disease in disease_list:
        if disease not in G:
            G.add_node(disease)
    
    node2vec = Node2Vec(G, dimensions=148, walk_length=80, num_walks=10, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    
    disease_list = [d for d in disease_list if d in model.wv]
    
    embeddings = np.array([model.wv[d] for d in disease_list])
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    shared_similarities, not_shared_similarities = calculate_similarity(disease_list, model, cosine)
    plot_similarity(shared_similarities, not_shared_similarities, 'Cosine Similarity', 'similarity.png')

driver.close()