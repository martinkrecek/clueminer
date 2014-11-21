package org.clueminer.partitioning.impl;

import java.util.ArrayList;
import java.util.LinkedList;
import org.clueminer.graph.api.Edge;
import org.clueminer.graph.api.Graph;
import org.clueminer.graph.api.Node;
import org.clueminer.partitioning.api.Bisection;

/**
 *
 * @author Tomas Bruna
 */
public class KernighanLin implements Bisection {

    private ArrayList<LinkedList<Vertex>> cluster;
    private Node[] nodes;
    private Vertex[] vertexes;
    private boolean[] used;
    private Vertex[] swapPair;
    private LinkedList<ArrayList<Vertex>> swapHistory;
    private LinkedList<Double> swapHistoryCost;
    private int nodeCount;
    private double maxCost;
    private int usedNodes;
    private Graph graph;

    public KernighanLin() {

    }
    
    public KernighanLin(Graph g) {
        graph = g;
    }


    @Override
    public ArrayList<LinkedList<Node>> bisect() {
        return bisect(graph);
    }
    
    @Override
    public ArrayList<LinkedList<Node>> bisect(Graph g) {
        initialize(g);

        createIntitalPartition();
        computeCosts();
        //nodeCount-1 for odd numbers of nodes
        while (usedNodes < nodeCount - 1) {
            findBestPair();
            swapPair();
            updateCosts();
        }
        swapUpToBestIndex();
        return createNodeClusters();
    }
    
    

     /**
     * Initialize all variables before bisection
     *
     * @param g graph to bisect
     */
    private void initialize(Graph g) {
        graph = g;
        nodes = g.getNodes().toArray();
        nodeCount = g.getNodeCount();
        usedNodes = 0;
        createVertexes();
        swapPair = new Vertex[2];
        swapHistory = new LinkedList<>();
        swapHistoryCost = new LinkedList<>();
    }

     /**
     * Create clusters of nodes form vertex array
     *
     * @return lists of nodes according to clusters
     */
    private ArrayList<LinkedList<Node>> createNodeClusters() {
        ArrayList<LinkedList<Node>> clusters = new ArrayList<>();
        clusters.add(new LinkedList<Node>());
        clusters.add(new LinkedList<Node>());
        for (Vertex vertex : vertexes) {
            if (vertex.cluster == 0) {
                clusters.get(0).add(nodes[vertex.index]);
            } else {
                clusters.get(1).add(nodes[vertex.index]);
            }
        }
        return clusters;
    }

     /**
     * Swap nodes form 0 up to maxDifferenceIndex.
     */
    private void swapUpToBestIndex() {
        int index = findBestSwaps();
        for (int i = 0; i <= index; i++) {
            int temp = swapHistory.get(i).get(0).cluster;
            swapHistory.get(i).get(0).cluster = swapHistory.get(i).get(1).cluster;
            swapHistory.get(i).get(1).cluster = temp;
        }

    }
    
     /**
     * Find how many swaps should be done to achieve best difference sum.
     */
    private int findBestSwaps() {
        int maxDifference = 0;
        int differenceSum = 0;
        int maxDifferenceIndex = -1;
        for (int i = 0; i < swapHistoryCost.size(); i++) {
            differenceSum += swapHistoryCost.get(i);
            if (differenceSum > maxDifference) {
                maxDifference = differenceSum;
                maxDifferenceIndex = i;
            }
        }
        return maxDifferenceIndex;
    }

     /**
     * Simulate swapping of two nodes. 
     * Real swapping is not done, nodes will be swapped at the end according to 
     * best difference sum
     */
    private void swapPair() {
        swapPair[0].used = true;
        swapPair[1].used = true;
        usedNodes += 2;
        ArrayList a = new ArrayList<>();
        a.add(swapPair[0]);
        a.add(swapPair[1]);
        swapHistoryCost.add(maxCost);
        swapHistory.add(a);
    }

     
     /**
     * Update differences of all nodes according to swapped pair. 
     */
    private void updateCosts() {
        for (int i = 0; i <= 1; i++) {
            ArrayList<Node> neighbors = (ArrayList<Node>) graph.getNeighbors(nodes[swapPair[i].index]).toCollection();
            for (Node neighbor : neighbors) {
                if (vertexes[graph.getIndex(neighbor)].cluster == swapPair[i].cluster) {
                    vertexes[graph.getIndex(neighbor)].difference += 2 * graph.getEdge(nodes[swapPair[i].index], neighbor).getWeight();
                } else {
                    vertexes[graph.getIndex(neighbor)].difference -= 2 * graph.getEdge(nodes[swapPair[i].index], neighbor).getWeight();
                }
            }
        }
    }

     /**
     * Compute differences of all nodes.
     */
    private void computeCosts() {
        for (Node node : nodes) {
            ArrayList<Node> neighbors = (ArrayList<Node>) graph.getNeighbors(node).toCollection();
            for (Node neighbor : neighbors) {
                if (vertexes[graph.getIndex(node)].cluster == vertexes[graph.getIndex(neighbor)].cluster) {
                    vertexes[graph.getIndex(node)].internalCost += graph.getEdge(node, neighbor).getWeight();
                } else {
                    vertexes[graph.getIndex(node)].externalCost += graph.getEdge(node, neighbor).getWeight();
                }

            }
            vertexes[graph.getIndex(node)].difference = vertexes[graph.getIndex(node)].externalCost - vertexes[graph.getIndex(node)].internalCost;
        }
    }

     /**
     * Find pair of nodes which has the highest sum of differences.
     */
    private void findBestPair() {
        maxCost = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < nodeCount; i++) {
            for (int j = i + 1; j < nodeCount; j++) {
                //having vertexes in list is pointless - sum would be n^2/2 too
                if ((vertexes[i].used || vertexes[j].used) || (vertexes[i].cluster == vertexes[j].cluster)) {
                    continue;
                }
                double edgeWeight;
                if (graph.getEdge(nodes[i], nodes[j]) != null) {
                    edgeWeight = graph.getEdge(nodes[i], nodes[j]).getWeight();
                } else {
                    edgeWeight = 0;
                }
                double cost = vertexes[i].difference + vertexes[j].difference - 2 * edgeWeight;
                if (cost > maxCost) {
                    maxCost = cost;
                    swapPair[0] = vertexes[i];
                    swapPair[1] = vertexes[j];
                }
            }
        }
    }

     /**
     * Create vertexes from nodes in graph.
     */
    private void createVertexes() {
        vertexes = new Vertex[nodeCount];
        for (int i = 0; i < nodeCount; i++) {
            vertexes[i] = new Vertex();
            vertexes[i].index = graph.getIndex(nodes[i]);
        }

    }

     /**
     * Randomly assign nodes to clusters at the beginning.
     */
    private void createIntitalPartition() {
        cluster = new ArrayList<>(2);
        cluster.add(new LinkedList<Vertex>());
        cluster.add(new LinkedList<Vertex>());
        for (int i = 0; i < nodeCount / 2; i++) {
            //  cluster.get(0).add(vertexes[i]);
            vertexes[i].cluster = 0;
        }
        for (int i = nodeCount / 2; i < nodeCount; i++) {
            //  cluster.get(1).add(vertexes[i]);
            vertexes[i].cluster = 1;
        }
    }

    public void printClusters() {
        ArrayList<LinkedList<Node>> clusters = createNodeClusters();
        for (int i = 0; i <= 1; i++) {
            System.out.print("Cluster " + i + ": ");
            for (Node n : clusters.get(i)) {
                System.out.print(n.getId() + ", ");
            }
            System.out.println("");
        }
    }

    @Override
    public Graph removeUnusedEdges() {
        for (int i = 0; i < nodeCount; i++) {
            for (int j = 0; j < nodeCount; j++) {
                if (vertexes[i].cluster != vertexes[j].cluster) {
                    Edge e = graph.getEdge(nodes[i], nodes[j]);
                    if (e != null) {
                        graph.removeEdge(e);
                    }
                }
            }
        }
        return graph; // deep copy or new graph needed
    }

    public class Vertex {

        public Vertex() {
            internalCost = externalCost = 0;
            used = false;
        }
        //reference to node not to have node array and vertex array
        int index;
        int cluster;
        boolean used;
        double internalCost;
        double externalCost;
        double difference;
    }
}