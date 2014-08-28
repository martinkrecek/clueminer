package org.clueminer.evolution.hac;

import java.util.List;
import java.util.Random;
import org.clueminer.clustering.api.AgglomerativeClustering;
import org.clueminer.clustering.api.Cluster;
import org.clueminer.clustering.api.Clustering;
import org.clueminer.clustering.api.HierarchicalResult;
import org.clueminer.clustering.api.evolution.Evolution;
import org.clueminer.clustering.api.evolution.Individual;
import org.clueminer.clustering.struct.DendrogramData;
import org.clueminer.distance.api.AbstractDistance;
import org.clueminer.distance.api.DistanceFactory;
import org.clueminer.eval.hclust.NaiveCutoff;
import org.clueminer.evolution.AbstractIndividual;
import org.clueminer.math.Matrix;
import org.clueminer.std.Scaler;
import org.clueminer.utils.Dump;
import org.clueminer.utils.Props;

/**
 *
 * @author Tomas Barton
 */
public class HclIndividual extends AbstractIndividual<HclIndividual> {

    private double fitness = 0;
    private static Random rand = new Random();
    private double[] weights;
    private Props params;
    private boolean debug = true;
    private Clustering clustering;

    public HclIndividual(Evolution evolution) {
        this.evolution = evolution;
        this.algorithm = evolution.getAlgorithm();
        init();
    }

    private void init() {
        weights = new double[evolution.attributesCount()];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = rand.nextDouble();
        }
        params = new Props();
        params.put("name", "HCL");
        // alg type
        params.put("alg-type", "cluster");
        // output class
        params.put("output-class", "single-output");
        params.putInt("distance-factor", 1);
        params.putBoolean("hcl-distance-absolute", true);

        String standard = "Min-Max";
        params.put("std", standard);
        params.putBoolean("fitted-params", false);

        /**
         * 0 for ALC method, 1 for CLC or -1 otherwise
         */
        int linkage = 1;
        /* if (radioLinkageAverage.isSelected()) {
         linkage = 0;
         } else if (radioLinkageComplete.isSelected()) {
         linkage = 1;
         } else if (radioLinkageSingle.isSelected()) {
         linkage = -1;
         }*/
        params.putInt("method-linkage", linkage);

        params.putBoolean("calculate-experiments", false);

        params.putBoolean("optimize-rows-ordering", true);

        //Clustering by Samples
        params.putBoolean("calculate-rows", true);
        //data.addParam("calculate-genes", String.valueOf(false));
        params.putBoolean("optimize-cols-ordering", true);

        params.putBoolean("optimize-sample-ordering", true);

        params.put("cutoff", "BIC score");
        params.putBoolean("log-scale", false);

    }

    @Override
    public void countFitness() {

        DistanceFactory df = DistanceFactory.getInstance();
        AbstractDistance func = df.getProvider("Euclidean");
        algorithm.setDistanceFunction(func);

        long start = System.currentTimeMillis();

        Matrix input = Scaler.standartize(evolution.getDataset().arrayCopy(), params.get("std", Scaler.NONE), params.getBoolean("log-scale", false));

        System.out.println("input matrix");
        input.print(5, 2);

        //   progress.setTitle("Clustering by rows");
        params.putBoolean("calculate-rows", true);
        HierarchicalResult rowsResult = ((AgglomerativeClustering) algorithm).hierarchy(input, evolution.getDataset(), params);
        Dump.array(rowsResult.getMapping(), "row mapping: ");

        //   progress.setTitle("Clustering by columns");
        //  params.setProperty("calculate-rows", String.valueOf(false));
        //  HierarchicalResult columnsResult = algorithm.hierarchy(input, evolution.getDataset(), params);
        // validate(columnsResult);
        //System.out.println("params: " + params.toString());
        //printResult(rowsResult);
        if (debug) {
            Matrix proximity = rowsResult.getProximityMatrix();
            //    Matrix cprox = columnsResult.getProximityMatrix();
            System.out.println("row proximity matrix:");
            proximity.print(5, 2);
            System.out.println("columns proximity matrix:");
            //  cprox.print(5, 2);
        }
        long time = System.currentTimeMillis() - start;
        System.out.println("clustering took " + time + " ms");

        //  double cutoff = rowsResult.findCutoff();
        //  System.out.println("rows tree cutoff = " + cutoff);
        //   cutoff = columnsResult.findCutoff();
        //   System.out.println("columns tree cutoff = " + cutoff);
        String cutoffAlg = params.get("cutoff", "NaiveCutoff");

        // if (!cutoffAlg.equals("-- naive --")) {
        //ClusterEvaluator eval = InternalEvaluatorFactory.getInstance().getProvider(cutoffAlg);
        NaiveCutoff strategy = new NaiveCutoff();
        rowsResult.findCutoff(strategy);

        // }// else we use a naive approach
        clustering = rowsResult.getClustering();

        DendrogramData dendroData = new DendrogramData(evolution.getDataset(), input, rowsResult);

        clustering.setParams(params);
        //clust.lookupAdd(evolution.getDataset());
        clustering.lookupAdd(dendroData);

        fitness = evolution.getEvaluator().score(clustering, evolution.getDataset());

    }

    @Override
    public double getFitness() {
        return fitness;
    }

    @Override
    public void mutate() {
        for (int i = 0; i < weights.length; i++) {
            if (rand.nextDouble() < evolution.getMutationProbability()) {
                weights[i] += rand.nextDouble();
            }
        }
    }

    /*  public List<Individual> cross(WeightsIndividual i) {
     ArrayList<Individual> offsprings = new ArrayList<Individual>();
     // we'll work with copies
     WeightsIndividual thisOne = this.deepCopy();
     WeightsIndividual secondOne = ((WeightsIndividual) i).deepCopy();
     int cross_id = rand.nextInt(evolution.attributesCount());
     System.arraycopy(((WeightsIndividual) i).weights, 0, thisOne.weights, 0, cross_id);
     System.arraycopy(((WeightsIndividual) i).weights, cross_id, secondOne.weights, cross_id, evolution.attributesCount() - cross_id);
     System.arraycopy(this.weights, 0, secondOne.weights, 0, cross_id);
     System.arraycopy(this.weights, cross_id, thisOne.weights, cross_id, evolution.attributesCount() - cross_id);
     offsprings.add(thisOne);
     offsprings.add(secondOne);
     return offsprings;
     }

     public HclIndividual deepCopy() {
     WeightsIndividual newOne = new WeightsIndividual(evolution);
     newOne.weights = new double[this.weights.length];
     System.arraycopy(this.weights, 0, newOne.weights, 0, this.weights.length);
     newOne.fitness = this.fitness;
     return newOne;
     }*/
    @Override
    public Clustering<Cluster> getClustering() {
        return clustering;
    }

    @Override
    public List<HclIndividual> cross(Individual other) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean isCompatible(Individual other) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public HclIndividual duplicate() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public HclIndividual deepCopy() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}