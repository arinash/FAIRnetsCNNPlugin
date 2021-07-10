import rdflib
from rdflib.namespace import Namespace
from rdflib.plugins.sparql import prepareQuery
import json

def sort_dict(dict, key):
    size = len(dict)
    for i in range(size):
        for j in range(size):
            if i == dict[j][key]:
                temp = dict[i]
                dict[i] = dict[j]
                dict[j] = temp
    return dict

if __name__ == "__main__":
    ontologyFile = r'.\documents\FAIRnets Dataset.ttl'

    #Load Graph
    print('Loading Graph')
    graph = rdflib.Graph()
    # Load Ontology
    graph.parse(ontologyFile, format="turtle")

    #Query to choose networks
    NetQuery = prepareQuery(
        """CONSTRUCT {
            ?network rdfs:label ?label .
            ?model nno:hasLayer ?layername .
            ?network nno:hasModel ?model.
            ?model nno:hasLossFunction ?lossfunc .
            ?model nno:hasOptimizer ?optname .
        } WHERE {
        SELECT ?label ?model ?layername ?lossfunc ?optname WHERE {
	        ?network a nno:NeuralNetwork .
	        ?network rdfs:label ?label .
	        ?network nno:hasModel ?model.
	        ?model nno:hasModelType ?modeltype.
	        ?network dc:description ?description .
            ?network nno:hasintendedUse ?use .
	        ?model nno:hasLayer ?layer.
            ?layer rdfs:label ?layername .
            ?model nno:hasLossFunction ?loss .
            ?loss rdfs:label ?lossfunc .
            ?model nno:hasOptimizer ?opt .
            ?opt rdfs:label ?optname .
            filter(!isBlank(?layer))
            filter(!isBlank(?layername))
            filter(!isBlank(?lossfunc))
            filter(!isBlank(?opt))
            filter(regex(lcase(str(?modeltype)), "cnn"))
            filter(regex(lcase(str(?use)), "images","i"))
            filter(regex(lcase(str(?description)), "image recognition","i")
                || regex(lcase(str(?description)), "recognize object in image","i")
                || regex(lcase(str(?description)), "recognize image","i")
                || regex(lcase(str(?description)), "image classification","i")
                || regex(lcase(str(?description)), "classify image","i")
                || regex(lcase(str(?description)), "digit recognition","i")
                || regex(lcase(str(?description)), "recognize digit","i")
                || regex(lcase(str(?description)), "face recognition","i")
                || regex(lcase(str(?description)), "recognize face","i")
                || regex(lcase(str(?description)), "face classification","i")
                || regex(lcase(str(?description)), "classify face","i")
                || regex(lcase(str(?description)), "identify image","i"))
        }} """,
        initNs={"nno": Namespace("https://w3id.org/nno/ontology#"),
                "rdfs": Namespace("http://www.w3.org/2000/01/rdf-schema#"),
                "rdf": Namespace("http://www.w3.org/1999/02/22-rdf-syntax-ns#"),
                "dc": Namespace("http://purl.org/dc/terms/")})

    result_graph_nets = graph.query(NetQuery)
    result_graph_nets.serialize(destination=r".\documents\nets.json", format='json-ld', indent=4)

    with open(r".\documents\nets.json", 'r', encoding="utf8") as f:
        nets_dict = json.load(f)

    #for net in nets_dict:
    for i in range(3):
        net_name = nets_dict[i]["@id"]
        lossfunc = nets_dict[i]["https://w3id.org/nno/ontology#hasLossFunction"][0]["@value"]
        optimizer = nets_dict[i]["https://w3id.org/nno/ontology#hasOptimizer"][0]["@value"]

        nets_dict[i]["Name"] = net_name
        nets_dict[i].pop("@id")
        nets_dict[i]["LossFunction"] = lossfunc
        nets_dict[i].pop("https://w3id.org/nno/ontology#hasLossFunction")
        nets_dict[i]["Optimizer"] = optimizer
        nets_dict[i].pop("https://w3id.org/nno/ontology#hasOptimizer")
        nets_dict[i]["Layers"] = nets_dict[i]["https://w3id.org/nno/ontology#hasLayer"]
        nets_dict[i].pop("https://w3id.org/nno/ontology#hasLayer")
        size = len(nets_dict[i]["Layers"])

        # Query to choose layers per network
        layerPerNetQuery = prepareQuery(
            """CONSTRUCT {
                ?model nno:hasLayer ?layer.
                ?layer rdfs:label ?layername .
                ?model nno:hasLayer ?layername .
                ?layer nno:hasLayerSequence ?layerseq .
                ?layer rdf:type ?typename.
            } WHERE {
            SELECT ?layer ?layername ?layerseq ?typename WHERE {
                ?network nno:hasModel ?model.
                ?model nno:hasLayer ?layer.
                ?layer rdfs:label ?layername .
                ?layer nno:hasLayerSequence ?layerseq .
                ?layer rdf:type ?type .
                ?type rdfs:label ?typename.
                filter(regex(lcase(str(?layer)), "%s" , "i"))
            }} ORDER BY asc(str(?layerseq))""" % net_name,
            initNs={"nno": Namespace("https://w3id.org/nno/ontology#"),
                    "rdfs": Namespace("http://www.w3.org/2000/01/rdf-schema#")})

        result_graph_layer = graph.query(layerPerNetQuery)
        result_graph_layer.serialize(destination=r".\documents\layers.json", format='json-ld', indent=4)

        with open(r".\documents\layers.json", 'r', encoding="utf8") as file:
            layers_dict = json.load(file)

        size = len(nets_dict[i]["Layers"])
        for j in range(size):
            for k in range(size):
                if nets_dict[i]["Layers"][k]["@value"] == \
                        layers_dict[j]["http://www.w3.org/2000/01/rdf-schema#label"][0]["@value"]:
                    nets_dict[i]["Layers"][k]["LayerType"] = \
                    layers_dict[j]["@type"][0]["@value"]
                    nets_dict[i]["Layers"][k]["sequence"] = \
                        layers_dict[j]["https://w3id.org/nno/ontology#hasLayerSequence"][0]["@value"]
        nets_dict[i]["Layers"] = sort_dict(nets_dict[i]["Layers"], "sequence")

    json_object = json.dumps(nets_dict, indent=4)
    with open(r".\documents\result.json", "w") as outfile:
       outfile.write(json_object)