# RareSim Master Project

# 1. Setup Project

## Download Ontologies
The ontologies are not stored direclty as it takes too much storage space.
To download the ontologies to you local file system, you can use the script in ontologies/model/load_models_to_local.


## Neo4J
For the knowledge graph, we are utilizing Neo4J. Start it via Docker

```
docker run -d --name neo4j-raresim \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/raresim123 \
  -e NEO4J_PLUGINS='["graph-data-science"]' \
  neo4j:5
```