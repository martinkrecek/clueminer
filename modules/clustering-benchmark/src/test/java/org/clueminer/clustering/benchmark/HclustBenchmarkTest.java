package org.clueminer.clustering.benchmark;

import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.clueminer.clustering.aggl.HAC;
import org.clueminer.clustering.aggl.HACLWMS;
import org.clueminer.clustering.api.AgglomerativeClustering;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.fixtures.clustering.FakeDatasets;
import org.clueminer.report.NanoBench;
import static org.junit.Assert.assertEquals;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 *
 * @author deric
 */
public class HclustBenchmarkTest {

    private final AgglomerativeClustering[] algorithms;

    public HclustBenchmarkTest() {
        //algorithms = new AgglomerativeClustering[]{new HAC(), new HACLW(), new HCL(), new HACLWMS()};
        algorithms = new AgglomerativeClustering[]{new HAC(), new HACLWMS()};
    }

    @BeforeClass
    public static void setUp() {
        Logger logger = NanoBench.getLogger();
        logger.setUseParentHandlers(false);
        logger.setLevel(Level.INFO);
        logger.addHandler(new ConsoleHandler());
    }

    @Test
    public void testSingleLinkage() {
        Dataset<? extends Instance> dataset = FakeDatasets.irisDataset();
        for (AgglomerativeClustering alg : algorithms) {
            NanoBench.create().measurements(2).cpuAndMemory().measure(
                    alg.getName() + " single link - " + dataset.getName(),
                    new HclustBenchmark().singleLinkage(alg, dataset)
            );
        }
    }

    @Test
    public void testCompleteLinkage() {
        Dataset<? extends Instance> dataset = FakeDatasets.irisDataset();
        for (AgglomerativeClustering alg : algorithms) {
            NanoBench.create().cpuAndMemory().measurements(2).measure(
                    alg.getName() + " complete link - " + dataset.getName(),
                    new HclustBenchmark().completeLinkage(alg, dataset)
            );
        }
    }

    /**
     * TODO: implement full tree diff
     */
    //@Test
    public void testSingleLinkageSameResult() {
        Dataset<? extends Instance> dataset = FakeDatasets.schoolData();
        //use one algorithm as reference one
        Container ref = new HclustBenchmark().singleLinkage(algorithms[0], dataset);
        ref.run();
        Container other;

        //compare result to others
        for (int i = 1; i < algorithms.length; i++) {
            AgglomerativeClustering algorithm = algorithms[i];
            other = new HclustBenchmark().singleLinkage(algorithm, dataset);
            other.run();
            System.out.println("comparing " + algorithms[0].getName() + " vs " + algorithm.getName());
            assertEquals(true, ref.equals(other));
        }
    }

}
