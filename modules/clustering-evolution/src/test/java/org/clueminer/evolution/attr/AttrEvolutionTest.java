package org.clueminer.evolution.attr;

import com.google.common.base.Supplier;
import com.google.common.collect.Maps;
import com.google.common.collect.Table;
import com.google.common.collect.Tables;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import org.clueminer.clustering.algorithm.KMeans;
import org.clueminer.clustering.api.Cluster;
import org.clueminer.clustering.api.ExternalEvaluator;
import org.clueminer.clustering.api.InternalEvaluator;
import org.clueminer.clustering.api.factory.InternalEvaluatorFactory;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.dataset.benchmark.ConsoleDump;
import org.clueminer.dataset.benchmark.DatasetFixture;
import org.clueminer.dataset.benchmark.GnuplotWriter;
import org.clueminer.dataset.benchmark.ResultsCollector;
import org.clueminer.eval.BIC;
import org.clueminer.eval.Silhouette;
import org.clueminer.eval.external.JaccardIndex;
import org.clueminer.utils.FileUtils;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import org.openide.util.NbBundle;

/**
 *
 * @author tombart
 */
public class AttrEvolutionTest {

    private static Dataset<Instance> irisDataset;
    private AttrEvolution test;
    //table for keeping results from experiments
    private final Table<String, String, Double> table;
    private static ResultsCollector rc;
    private static String benchmarkFolder;
    private static String csvOutput;

    public AttrEvolutionTest() {
        table = Tables.newCustomTable(
                Maps.<String, Map<String, Double>>newHashMap(),
                new Supplier<Map<String, Double>>() {
                    @Override
                    public Map<String, Double> get() {
                        return Maps.newHashMap();
                    }
                });

        String home = System.getProperty("user.home") + File.separatorChar
                + NbBundle.getMessage(
                        FileUtils.class,
                        "FOLDER_Home");
        createFolder(home);
        benchmarkFolder = home + File.separatorChar + "benchmark";
        createFolder(benchmarkFolder);
        rc = new ResultsCollector(table);
        csvOutput = benchmarkFolder + File.separatorChar + "results.csv";
    }

    private void createFolder(String folder) {
        File file = new File(folder);
        if (!file.exists()) {
            if (file.mkdir()) {
                System.out.println("Directory " + folder + " created!");
            } else {
                System.out.println("Failed to create " + folder + "directory!");
            }
        }
    }

    @BeforeClass
    public static void setUpClass() throws FileNotFoundException, IOException {
        irisDataset = DatasetFixture.iris();
    }

    @AfterClass
    public static void tearDownClass() {
        System.out.println("writing results to file: " + csvOutput);
        rc.writeToCsv(csvOutput);
    }

    /**
     * Test of attributesCount method, of class AttrEvolution.
     */
    @Test
    public void testAttributesCount() {
    }

    /**
     * Test of getDataset method, of class AttrEvolution.
     */
    @Test
    public void testGetDataset() {
    }

    /**
     * Test of run method, of class AttrEvolution.
     */
    //  @Test
    public void testRun() {
        test = new AttrEvolution(irisDataset, 50);
        test.setAlgorithm(new KMeans());
        test.setEvaluator(new BIC());
        ExternalEvaluator ext = new JaccardIndex();
        test.setExternal(ext);
        //collect data from evolution
        GnuplotWriter gw = new GnuplotWriter(test, benchmarkFolder, "iris-evolution");
        gw.setPlotDumpMod(1);
        test.addEvolutionListener(gw);
        //test.addEvolutionListener(new ConsoleDump(ext));
        //test.setEvaluator(new JaccardIndex());
        test.run();
    }

    // @Test
    public void testInformed() {
        //test run with informed metric
        test = new AttrEvolution(irisDataset, 50);
        test.setAlgorithm(new KMeans());
        ExternalEvaluator ext = new JaccardIndex();
        test.setEvaluator(ext);
        test.setExternal(ext);
        //collect data from evolution
        test.addEvolutionListener(new ConsoleDump());
        test.addEvolutionListener(new GnuplotWriter(test, benchmarkFolder, "iris-evolution-informed"));
        //test.setEvaluator(new JaccardIndex());
        test.run();
    }

    @Test
    public void testVariousMeasuresAndDatasets() {
        InternalEvaluatorFactory<Instance, Cluster<Instance>> factory = InternalEvaluatorFactory.getInstance();
        ExternalEvaluator ext = new JaccardIndex();
        Map<Dataset<Instance>, Integer> datasets = new HashMap<>();
        //just to make the test fast
        datasets.put(DatasetFixture.insect(), 3);

        String name;

        for (Entry<Dataset<Instance>, Integer> entry : datasets.entrySet()) {
            Dataset<Instance> dataset = entry.getKey();
            name = dataset.getName();
            System.out.println("=== dataset " + name);
            System.out.println("size: " + dataset.size());
            System.out.println(dataset.toString());
            String dataDir = benchmarkFolder + File.separatorChar + name;
            (new File(dataDir)).mkdir();
            for (InternalEvaluator eval : factory.getAll()) {
                System.out.println("evaluator: " + eval.getName());
                test = new AttrEvolution(dataset, 20);
                test.setAlgorithm(new KMeans());
                test.setK(entry.getValue());
                test.setEvaluator(eval);
                test.setExternal(ext);
                GnuplotWriter gw = new GnuplotWriter(test, benchmarkFolder, name + "/" + name + "-" + safeName(eval.getName()));
                gw.setPlotDumpMod(50);
                //collect data from evolution
                //test.addEvolutionListener(new ConsoleDump());
                test.addEvolutionListener(gw);
                test.addEvolutionListener(rc);
                test.run();
            }
            String csvRes = benchmarkFolder + File.separatorChar + name + File.separatorChar + name + ".csv";
            rc.writeToCsv(csvRes);

        }
    }

    //@Test
    public void testSilhouette() {
        ExternalEvaluator ext = new JaccardIndex();
        InternalEvaluator eval = new Silhouette();
        Dataset<Instance> dataset = DatasetFixture.glass();
        String name = dataset.getName();
        System.out.println("evaluator: " + eval.getName());
        test = new AttrEvolution(dataset, 20);
        test.setAlgorithm(new KMeans());
        test.setK(7);
        test.setEvaluator(eval);
        test.setExternal(ext);
        GnuplotWriter gw = new GnuplotWriter(test, benchmarkFolder, name + "/" + name + "-" + safeName(eval.getName()));
        gw.setPlotDumpMod(50);
        //collect data from evolution
        //test.addEvolutionListener(new ConsoleDump(ext));
        test.addEvolutionListener(gw);
        //test.setEvaluator(new JaccardIndex());
        test.run();
    }

    private String safeName(String name) {
        return name.toLowerCase().replace(" ", "_");
    }

    /**
     * Test of getMutationProbability method, of class AttrEvolution.
     */
    @Test
    public void testGetMutationProbability() {
    }

    /**
     * Test of setMutationProbability method, of class AttrEvolution.
     */
    @Test
    public void testSetMutationProbability() {
    }

    /**
     * Test of getCrossoverProbability method, of class AttrEvolution.
     */
    @Test
    public void testGetCrossoverProbability() {
    }

    /**
     * Test of setCrossoverProbability method, of class AttrEvolution.
     */
    @Test
    public void testSetCrossoverProbability() {
    }

}
