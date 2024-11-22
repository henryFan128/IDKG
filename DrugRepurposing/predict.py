import torch
import torch.nn.functional as F
from neo4j import GraphDatabase
from torch_geometric.data import Data

uri = "bolt://localhost:7687"
username = "neo4j"
password = "xxxxxxxx"
ModelPath = "./checkpoints/xxx.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(ModelPath, map_location=device)
model.to(device)

model.eval()

def ReadData(uri, username, password):
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        drugs = session.run("MATCH (d:Drug) RETURN elementId(d) as id, d.Drug_name as name").data()
        diseases = session.run("MATCH (d:Disease) RETURN elementId(d) as id, d.Disease_name as name").data()
    return drugs, diseases

def GeneratePotentialEdges(drugs, diseases, full_graph_data):
    potential_edges = []
    node_index = {node['id']: i for i, node in enumerate(drugs + diseases)}

    x = full_graph_data.x.to(device) 
    edge_index = full_graph_data.edge_index.to(device)  
    data = Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        z = model(data)  

    for drug in drugs:
        for disease in diseases:
            drug_index = node_index[drug['id']]
            disease_index = node_index[disease['id']]

            drug_embedding = z[drug_index]
            disease_embedding = z[disease_index]

            drug_embedding = F.normalize(drug_embedding, p=2, dim=0)
            disease_embedding = F.normalize(disease_embedding, p=2, dim=0)
            score = torch.sigmoid(torch.dot(drug_embedding, disease_embedding)).item()

            potential_edges.append((drug['name'], disease['name'], score))

    return potential_edges

def WriteTopEdgesToFile(potential_edges, filename, top_n=10):
    sorted_edges = sorted(potential_edges, key=lambda x: x[2], reverse=True)

    top_edges = sorted_edges[:top_n]

    with open(filename, 'w') as f:
        for edge in top_edges:
            f.write(f"Drug: {edge[0]}, Disease: {edge[1]}, Score: {edge[2]}\n")

if __name__ == "__main__":
    drugs, diseases = ReadData(uri, username, password)
    full_graph_data = torch.load('data.pth', map_location=device)  

    potential_edges = GeneratePotentialEdges(drugs, diseases, full_graph_data)
    WriteTopEdgesToFile(potential_edges, 'DrDisPred.txt')
