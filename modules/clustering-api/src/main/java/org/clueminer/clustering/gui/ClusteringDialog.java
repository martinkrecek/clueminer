package org.clueminer.clustering.gui;

import javax.swing.JPanel;
import org.clueminer.clustering.api.Cluster;
import org.clueminer.clustering.api.ClusteringAlgorithm;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.utils.Props;

/**
 * Each algorithm can have a GUI panel which should implement this interface
 *
 * @author Tomas Barton
 * @param <E>
 * @param <C>
 */
public interface ClusteringDialog<E extends Instance, C extends Cluster<E>> {

    /**
     * For lookup purposes, should be unique
     *
     * @return
     */
    String getName();

    /**
     * Parameters configured by user (or default ones)
     *
     * @return
     */
    Props getParams();

    /**
     * GUI which will be embedded into another dialog (should not contain any
     * OK/Cancel buttons)
     *
     * @return
     */
    JPanel getPanel();

    /**
     * Return true when UI is compatible with given algorithm
     *
     * @param algorithm
     * @param dataset in some cases dataset could be used for setting boundaries
     * of parameters
     * @return
     */
    boolean isUIfor(ClusteringAlgorithm<E, C> algorithm, Dataset<E> dataset);
}
