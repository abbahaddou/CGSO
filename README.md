# CGNN


## Spectral Clustering on Barabasi Albert Models



  - To generate the 200 synthetic Graphs :
    ```
    cd ./Spectral_clustering_BA/
    python generate_graphs.py
    ```
  - To run the spectral clustering on the 200 generated graphs using different centrality metrics:
    ```
    cd ./Spectral_clustering_BA/
    python spectral_clustering.py --centrality 'DEGREE'
    python spectral_clustering.py --centrality "KCORE"
    python spectral_clustering.py --centrality "PAGERANK"
    python spectral_clustering.py --centrality "PATHS"
    ```

## Spectral Clustering on Cora


  - To run the spectral clustering on Cora using different centrality metrics, we need to specify the exponenent values e1 and e1
  Example : e1 = -0.5 , e2 = -0.5
    ```
    cd ./Spectral_clustering_Cora/
    python spectral_clustering.py --centrality 'DEGREE' --e1 -0.5 --e2 -0.5
    python spectral_clustering.py --centrality "KCORE" --e1 -0.5 --e2 -0.5
    python spectral_clustering.py --centrality "PAGERANK" --e1 -0.5 --e2 -0.5
    python spectral_clustering.py --centrality "PATHS" --e1 -0.5 --e2 -0.5
    ```

## CGNN - Node Classification
    To run the node classification task, we need to specify 3 main inputs: 1/ The dataset (e.g. Cora), 2/ The used Centrality (e.g 'DEGREE', 'KCORE', 'PAGERANK' or 'PATHS') 
    and the initial values of the hyperparameters of the GSO (e.g, 'MEANAGG')
    ```
    python train_nodes.py --dataset "Cora" --centrality 'DEGREE' --init 'MEANAGG'
    ```

