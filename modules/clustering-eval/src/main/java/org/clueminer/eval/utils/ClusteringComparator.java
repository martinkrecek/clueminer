package org.clueminer.eval.utils;

import java.util.Comparator;
import org.clueminer.clustering.api.Cluster;
import org.clueminer.clustering.api.ClusterEvaluation;
import org.clueminer.clustering.api.Clustering;
import org.clueminer.clustering.api.EvaluationTable;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;

/**
 *
 * @author Tomas Barton
 */
public class ClusteringComparator implements Comparator<Clustering> {

    private ClusterEvaluation evaluator;
    private boolean asc = true;

    public ClusteringComparator() {

    }

    public ClusteringComparator(ClusterEvaluation eval) {
        this.evaluator = eval;
    }

    @Override
    public int compare(Clustering c1, Clustering c2) {
        EvaluationTable t1 = evaluationTable(c1);
        EvaluationTable t2 = evaluationTable(c2);

        double s1 = t1.getScore(evaluator);
        double s2 = t2.getScore(evaluator);
        boolean bigger;

        if (s1 == s2) {
            return 0;
        }
        bigger = evaluator.isBetter(s1, s2);

        if (!asc) {
            bigger = !bigger;
        }
        // "best" solution is at the end
        if (bigger) {
            return 1;
        } else {
            return -1;
        }
    }

    protected EvaluationTable evaluationTable(Clustering<? extends Cluster> clustering) {
        EvaluationTable evalTable = clustering.getLookup().lookup(EvaluationTable.class);
        //we try to compute score just once, to eliminate delays
        if (evalTable == null) {
            Dataset<? extends Instance> dataset = clustering.getLookup().lookup(Dataset.class);
            if (dataset == null) {
                throw new RuntimeException("no dataset in lookup");
            }
            evalTable = new HashEvaluationTable(clustering, dataset);
            clustering.lookupAdd(evalTable);
        }
        return evalTable;
    }

    public ClusterEvaluation getEvaluator() {
        return evaluator;
    }

    public void setEvaluator(ClusterEvaluation evaluator) {
        this.evaluator = evaluator;
    }

    public boolean isAsc() {
        return asc;
    }

    public void setAsc(boolean asc) {
        this.asc = asc;
    }

}
