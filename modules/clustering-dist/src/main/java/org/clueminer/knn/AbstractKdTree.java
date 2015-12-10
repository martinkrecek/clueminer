/*
 * Copyright (C) 2011-2015 clueminer.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package org.clueminer.knn;

import java.util.List;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.distance.EuclideanDistance;
import org.clueminer.neighbor.KNNSearch;
import org.clueminer.neighbor.NearestNeighborSearch;
import org.clueminer.neighbor.Neighbor;
import org.clueminer.neighbor.RNNSearch;
import org.clueminer.sort.HeapSelect;
import org.clueminer.utils.Props;

/**
 *
 * @author deric
 * @param <E>
 */
public abstract class AbstractKdTree<T> {

    public static final String name = "KD-tree";

    // All types
    private final int dimensions;
    private final AbstractKdTree<T> parent;

    // Root only
    private final LinkedList<double[]> locationStack;
    private final Integer sizeLimit;

    // Leaf only
    private double[][] locations;
    private Object[] data;
    private int locationCount;

    // Stem only
    private AbstractKdTree<T> left, right;
    private int splitDimension;
    private double splitValue;

    // Bounds
    private double[] minLimit, maxLimit;
    private boolean singularity;

    // Temporary
    private Status status;

    /**
     * The root node of KD-Tree.
     */
    protected AbstractKdTree(int dimensions, Integer sizeLimit) {
        this.dimensions = dimensions;

    /**
     * Constructor.
     *
     * @param dataset
     */
    protected AbstractKdTree(AbstractKdTree<T> parent, boolean right) {
        this.dimensions = parent.dimensions;

    public KDTree() {
        this.dm = new EuclideanDistance();
        ((EuclideanDistance) dm).setSqrt(false);
    }

    private void buildTree() {
        if (dataset == null) {
            throw new RuntimeException("missing dataset");
        }
        if (dataset.isEmpty()) {
            throw new RuntimeException("can't build kd-tree from an empty dataset");
        }
        if (dataset.attributeCount() == 0) {
            throw new RuntimeException("Dataset doesn't have any attributes");
        }
        int n = dataset.size();
        index = new int[n];
        for (int i = 0; i < n; i++) {
            index[i] = i;
        }
        // Build the tree
        root = buildNode(0, n);
    }

    @Override
    public String getName() {
        return name;
    }

    /**
     * Build a k-d tree from the given set of dataset.
     */
    public void addPoint(double[] location, T value) {
        AbstractKdTree<T> cursor = this;

        // Allocate the node
        KDNode node = new KDNode();

        // Fill in basic info
        node.count = end - begin;
        node.index = begin;

        // Calculate the bounding box
        double[] lowerBound = new double[d];
        double[] upperBound = new double[d];

        for (int i = 0; i < d; i++) {
            lowerBound[i] = dataset.get(index[begin], i);
            upperBound[i] = dataset.get(index[begin], i);
        }

                // Create child leaves
                AbstractKdTree<T> left = new ChildNode(cursor, false);
                AbstractKdTree<T> right = new ChildNode(cursor, true);

                // Move locations into children
                for (int i = 0; i < cursor.locationCount; i++) {
                    double[] oldLocation = cursor.locations[i];
                    Object oldData = cursor.data[i];
                    if (oldLocation[cursor.splitDimension] > cursor.splitValue) {
                        // Right
                        right.locations[right.locationCount] = oldLocation;
                        right.data[right.locationCount] = oldData;
                        right.locationCount++;
                        right.extendBounds(oldLocation);
                    } else {
                        // Left
                        left.locations[left.locationCount] = oldLocation;
                        left.data[left.locationCount] = oldData;
                        left.locationCount++;
                        left.extendBounds(oldLocation);
                    }
                }
                if (upperBound[j] < c) {
                    upperBound[j] = c;
                }
            }
        }

        // Calculate bounding box stats
        double maxRadius = -1;
        for (int i = 0; i < d; i++) {
            double radius = (upperBound[i] - lowerBound[i]) / 2;
            if (radius > maxRadius) {
                maxRadius = radius;
                node.split = i;
                node.cutoff = (upperBound[i] + lowerBound[i]) / 2;
            }
        }

        // If the max spread is 0, make this a leaf node
        if (maxRadius == 0) {
            node.lower = node.upper = null;
            return node;
        }

        // Partition the dataset around the midpoint in this dimension. The
        // partitioning is done in-place by iterating from left-to-right and
        // right-to-left in the same way that partioning is done in quicksort.
        int i1 = begin, i2 = end - 1, size = 0;
        while (i1 <= i2) {
            boolean i1Good = (dataset.get(index[i1], node.split) < node.cutoff);
            boolean i2Good = (dataset.get(index[i2], node.split) >= node.cutoff);

            if (!i1Good && !i2Good) {
                int temp = index[i1];
                index[i1] = index[i2];
                index[i2] = temp;
                i1Good = i2Good = true;
            }

    /**
     * Remove the oldest value from the tree. Note: This cannot trim the bounds
     * of nodes, nor empty nodes, and thus you can't expect it to perfectly
     * preserve the speed of the tree as you keep adding.
     */
    private void removeOld() {
        double[] location = this.locationStack.removeFirst();
        AbstractKdTree<T> cursor = this;

        // Find the node where the point is
        while (cursor.locations == null) {
            if (location[cursor.splitDimension] > cursor.splitValue) {
                cursor = cursor.right;
            } else {
                cursor = cursor.left;
            }

            if (i2Good) {
                i2--;
            }
        }

        // Create the child nodes
        node.lower = buildNode(begin, begin + size);
        node.upper = buildNode(begin + size, end);

        return node;
    }

    /**
     * Returns the nearest neighbors of the given target starting from the give
     * tree node.
     *
     * @param q the query key.
     * @param node the root of subtree.
     * @param neighbor the current nearest neighbor.
     */
    @SuppressWarnings("unchecked")
    public List<Entry<T>> nearestNeighbor(double[] location, int count, boolean sequentialSorting) {
        AbstractKdTree<T> cursor = this;
        cursor.status = Status.NONE;
        double range = Double.POSITIVE_INFINITY;
        ResultHeap resultHeap = new ResultHeap(count);

        do {
            if (cursor.status == Status.ALLVISITED) {
                // At a fully visited part. Move up the tree
                cursor = cursor.parent;
                continue;
            }

            if (cursor.status == Status.NONE && cursor.locations != null) {
                // At a leaf. Use the data.
                if (cursor.locationCount > 0) {
                    if (cursor.singularity) {
                        double dist = pointDist(cursor.locations[0], location);
                        if (dist <= range) {
                            for (int i = 0; i < cursor.locationCount; i++) {
                                resultHeap.addValue(dist, cursor.data[i]);
                            }
                        }
                    } else {
                        for (int i = 0; i < cursor.locationCount; i++) {
                            double dist = pointDist(cursor.locations[i], location);
                            resultHeap.addValue(dist, cursor.data[i]);
                        }
                    }
                    range = resultHeap.getMaxDist();
                }
                //TODO: squared distance would be enough
                double distance = dm.measure(q, dataset.get(index[idx]));
                if (distance < neighbor.distance) {
                    neighbor.key = dataset.get(index[idx]);
                    neighbor.index = index[idx];
                    neighbor.distance = distance;
                }
            }

            // Going to descend
            AbstractKdTree<T> nextCursor = null;
            if (cursor.status == Status.NONE) {
                // At a fresh node, descend the most probably useful direction
                if (location[cursor.splitDimension] > cursor.splitValue) {
                    // Descend right
                    nextCursor = cursor.right;
                    cursor.status = Status.RIGHTVISITED;
                } else {
                    // Descend left;
                    nextCursor = cursor.left;
                    cursor.status = Status.LEFTVISITED;
                }
            } else if (cursor.status == Status.LEFTVISITED) {
                // Left node visited, descend right.
                nextCursor = cursor.right;
                cursor.status = Status.ALLVISITED;
            } else if (cursor.status == Status.RIGHTVISITED) {
                // Right node visited, descend left.
                nextCursor = cursor.left;
                cursor.status = Status.ALLVISITED;
            }

            search(q, nearer, neighbor);

            // now look in further half
            if (neighbor.distance >= diff * diff) {
                search(q, further, neighbor);
            }
        }
    }

    /**
     * Returns (in the supplied heap object) the k nearest neighbors of the
     * given target starting from the give tree node.
     *
     * @param q the query key.
     * @param node the root of subtree.
     * @param k the number of neighbors to find.
     * @param heap the heap object to store/update the kNNs found during the
     * search.
     */
    private class ChildNode extends AbstractKdTree<T> {

        private ChildNode(AbstractKdTree<T> parent, boolean right) {
            super(parent, right);
        }

        // Distance measurements are always called from the root node
        protected double pointDist(double[] p1, double[] p2) {
            throw new IllegalStateException();
        }

        protected double pointRegionDist(double[] point, double[] min, double[] max) {
            throw new IllegalStateException();
        }
    }

    /**
     * Class for tree with Weighted Squared Euclidean distancing
     */
    public static class WeightedSqrEuclid<T> extends AbstractKdTree<T> {

        private double[] weights;

        public WeightedSqrEuclid(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
            this.weights = new double[dimensions];
            Arrays.fill(this.weights, 1.0);
        }

        public void setWeights(double[] weights) {
            this.weights = weights;
        }

        protected double getAxisWeightHint(int i) {
            return weights[i];
        }

        protected double pointDist(double[] p1, double[] p2) {
            double d = 0;

            for (int i = 0; i < p1.length; i++) {
                double diff = (p1[i] - p2[i]) * weights[i];
                if (!Double.isNaN(diff)) {
                    d += diff * diff;
                }
            }
        } else {
            KDNode nearer, further;
            double diff = q.get(node.split) - node.cutoff;
            if (diff < 0) {
                nearer = node.lower;
                further = node.upper;
            } else {
                nearer = node.upper;
                further = node.lower;
            }

            search(q, nearer, heap);

            // now look in further half
            if (heap.peek().distance >= diff * diff) {
                search(q, further, heap);
            }
        }
    }

    /**
     * Returns the neighbors in the given range of search target from the give
     * tree node.
     *
     * @param q the query key.
     * @param node the root of subtree.
     * @param radius	the radius of search range from target.
     * @param neighbors the list of found neighbors in the range.
     */
    public static class SqrEuclid<T> extends AbstractKdTree<T> {

                double distance = dm.measure(q, dataset.get(index[idx]));
                if (distance <= radius) {
                    neighbors.add(new Neighbor<>(dataset.get(index[idx]), index[idx], distance));
                }
            }
        } else {
            KDNode nearer, further;
            double diff = q.get(node.split) - node.cutoff;
            if (diff < 0) {
                nearer = node.lower;
                further = node.upper;
            } else {
                nearer = node.upper;
                further = node.lower;
            }

            search(q, nearer, radius, neighbors);

            // now look in further half
            if (radius >= diff * diff) {
                search(q, further, radius, neighbors);
            }
        }
    }

    /**
     * Class for tree with Weighted Manhattan distancing
     */
    public static class WeightedManhattan<T> extends AbstractKdTree<T> {

    @Override
    public Neighbor<E>[] knn(E q, int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("Invalid k: " + k);
        }

        if (k > dataset.size()) {
            throw new IllegalArgumentException("Neighbor array length is larger than the dataset size");
        }

        Neighbor<E> neighbor = new Neighbor<>(null, 0, Double.MAX_VALUE);
        @SuppressWarnings("unchecked")
        Neighbor<E>[] neighbors = (Neighbor<E>[]) java.lang.reflect.Array.newInstance(neighbor.getClass(), k);
        HeapSelect<Neighbor<E>> heap = new HeapSelect<>(neighbors);
        for (int i = 0; i < k; i++) {
            heap.add(neighbor);
            neighbor = new Neighbor<>(null, 0, Double.MAX_VALUE);
        }

        search(q, root, heap);
        heap.sort();
        for (Neighbor<E> neighbor1 : neighbors) {
            neighbor1.distance = Math.sqrt(neighbor1.distance);
        }

        return neighbors;
    }

    /**
     * Class for tree with Manhattan distancing
     */
    public static class Manhattan<T> extends AbstractKdTree<T> {

        public Manhattan(int dimensions, Integer sizeLimit) {
            super(dimensions, sizeLimit);
        }

        search(q, root, radius, neighbors);
    }

    @Override
    public Neighbor[] knn(E q, int k, Props params) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void setDataset(Dataset<E> dataset) {
        this.dataset = dataset;
        buildTree();
    }

    public void delete(E q) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void insert(E q, int index) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public String toString() {
        return "KD-Tree";
    }

}
