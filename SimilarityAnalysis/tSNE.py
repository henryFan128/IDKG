from matplotlib.font_manager import FontProperties
from neo4j import GraphDatabase
import networkx as nx
from node2vec import Node2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random

# 设置 Arial 字体路径
font_path = './Arial.ttf'
arial_font = FontProperties(fname=font_path)
arial_bold_font = FontProperties(fname=font_path, weight='bold')

""" uri = "bolt://localhost:7687"  
username = "neo4j"  
password = "xxxxxxxxxxx"      """

uri = "Neo4j://219.228.149.80:7687"  # 根据实际情况修改
username = "neo4j"              # 根据实际情况修改
password = "emotion-gilbert-rhino-orient-gyro-7433"     # 根据实际情况修改

driver = GraphDatabase.driver(uri, auth=(username, password))

def get_data():
    with driver.session() as session:
        """ 
        Node ID; Node name; node label 
        """

        node_result = session.run("""
            MATCH (n)
            WHERE n:Drug OR n:Disease OR n:Protein OR n:Pathogen OR n:Pathway OR n:Gene OR n:ADRs OR n:Small_Molecule
            RETURN id(n) AS id, labels(n) AS labels,
                CASE
                    WHEN n:Drug THEN n.Drug_name
                    WHEN n:Disease THEN n.Disease_name
                    WHEN n:Protein THEN n.Protein_name
                    WHEN n:Pathogen THEN n.Pathogen_name
                    WHEN n:Pathway THEN n.Pathway_name
                    WHEN n:Gene THEN n.GeneSymbol
                    WHEN n:ADRs THEN n.Symptom_name
                    WHEN n:Small_Molecule THEN n.Small_Molecule_name
                    ELSE n.name
                END AS name
        """)
        node_types = {'Drug': [], 'Disease': [], 'Protein': [], 'Pathogen': [], 'Pathway': [], 'Gene': [], 'ADRs': [], 'Small_Molecule': []}
        node_info = {}  
        for record in node_result:
            node_id = record["id"]
            labels = record["labels"]  
            name = record.get("name", "")
            for label in labels:
                if label in node_types:
                    node_types[label].append(node_id)
                    node_info[node_id] = {
                        'type': label,
                        'name': name
                    }
                    break  
        sampled_nodes = []
        for node_type, ids in node_types.items():
            if len(ids) > 150:
                sampled = random.sample(ids, 150)
            else:
                sampled = ids
            sampled_nodes.extend(sampled)

        edge_result = session.run("""
            MATCH (n)-[r]-(m)
            WHERE id(n) IN $node_ids AND id(m) IN $node_ids
            RETURN id(n) AS source, id(m) AS target
        """, node_ids=sampled_nodes)
        edges = [(record["source"], record["target"]) for record in edge_result]

        node_info = {node_id: node_info[node_id] for node_id in sampled_nodes}

    return sampled_nodes, edges, node_info

nodes, edges, node_info = get_data()
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

embedding_dimensions = 148
node2vec = Node2Vec(G, dimensions=embedding_dimensions, walk_length=80, num_walks=50, workers=8)
model = node2vec.fit(window=10, min_count=1)

embeddings = model.wv
node_ids = model.wv.index_to_key  

X = np.array([embeddings[node_id] for node_id in node_ids])  

tsne = TSNE(n_components=2, random_state=7, perplexity=10)
X_tsne = tsne.fit_transform(X)

tsne_df = pd.DataFrame(X_tsne, columns=['x', 'y'])
tsne_df['type'] = [node_info[int(node_id)]['type'] for node_id in node_ids]

plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")  
sns.scatterplot(data=tsne_df, x='x', y='y', hue='type', palette={
    'Drug': '#DE7833',
    'Disease': '#912C2C',
    'Protein': '#F2BB6B',
    'Pathogen': '#C2ABC8',
    'Pathway': '#329845', 
    'Gene': '#276C9E',
    'ADRs': '#AED185',
    'Small_Molecule': '#A3C9D5'
})
plt.legend(prop=arial_bold_font)
# plt.xlabel('dimension 1', fontsize=14, fontproperties=arial_font)
# plt.ylabel('dimension 2', fontsize=14, fontproperties=arial_font)
plt.savefig('tsne3', dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

driver.close()