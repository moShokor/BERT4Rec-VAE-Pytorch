import pickle as pk
from fastnode2vec import Graph, Node2Vec

with open('./graph.pk', 'rb') as f:
    graph = pk.load(f)

train = True

if train:

    # training undirected
    graph = Graph(graph, directed=False, weighted=False)

    # n2v = Node2Vec(graph, dim=100, walk_length=100, context=16, p=2.0, q=0.5, workers=4, hs=1)
    n2v = Node2Vec(graph, dim=500, walk_length=100, context=16, p=0.25, q=0.25, workers=8, hs=1)

    n2v.train(epochs=100)

    with open('./graph_embed_directed_HS1_unristricted_d500.pk', 'wb') as f:
        pk.dump(n2v, f, protocol=4)
