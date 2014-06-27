package org.clueminer.clustering.aggl;

import java.io.IOException;
import java.util.prefs.Preferences;
import org.clueminer.clustering.api.HierarchicalResult;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.dataset.plugin.ArrayDataset;
import org.clueminer.fixtures.CommonFixture;
import org.clueminer.math.Matrix;
import org.clueminer.math.matrix.JMatrix;
import org.clueminer.hclust.linkage.SingleLinkage;
import org.clueminer.io.CsvLoader;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;
import org.openide.util.Exceptions;
import org.openide.util.NbPreferences;

/**
 *
 * @author deric
 */
public class HACTest {

    private static final HAC subject = new HAC();
    private static Dataset<? extends Instance> school;
    private static final CommonFixture fixture = new CommonFixture();

    public HACTest() {
    }

    @BeforeClass
    public static void setUpClass() {

    }

    @AfterClass
    public static void tearDownClass() {
    }

    @Before
    public void setUp() {
    }

    @After
    public void tearDown() {
    }

    /**
     * Test of getName method, of class HC1.
     */
    @Test
    public void testGetName() {
        assertNotNull(subject.getName());
    }

    /**
     * Test of cluster method, of class HC1.
     */
    @Test
    public void testCluster_Matrix_Preferences() {
    }

    /**
     * Test of hierarchy method, of class HC1.
     */
    @Test
    public void testHierarchy_Dataset_Preferences() {
    }

    public static Dataset<? extends Instance> schoolData() {
        if (school == null) {
            CsvLoader loader = new CsvLoader();
            school = new ArrayDataset(17, 4);
            loader.setClassIndex(4);
            loader.setSeparator(' ');
            try {
                loader.load(fixture.schoolData(), school);
            } catch (IOException ex) {
                Exceptions.printStackTrace(ex);
            }
        }
        return school;

    }

    /**
     * Test of hierarchy method, of class HC1.
     *
     * @throws java.io.IOException
     */
    @Test
    public void testHierarchy_3args() throws IOException {
        Dataset<? extends Instance> dataset = schoolData();
        assertEquals(17, dataset.size());
        assertEquals(4, dataset.attributeCount());
        Matrix input = new JMatrix(dataset.arrayCopy());
        input.print(3, 2);
        Preferences pref = NbPreferences.forModule(HACTest.class);
        pref.put(AgglParams.LINKAGE, SingleLinkage.name);
        pref.putBoolean(AgglParams.CLUSTER_ROWS, true);
        HierarchicalResult result = subject.hierarchy(input, dataset, pref);
        Matrix similarityMatrix = result.getProximityMatrix();
        assertNotNull(similarityMatrix);
        assertEquals(similarityMatrix.rowsCount(), dataset.size());
        assertEquals(similarityMatrix.columnsCount(), dataset.size());

        similarityMatrix.printLower(5, 2);
        result.getTreeData().print();
    }

    @Test
    public void testColumnClustering() throws IOException {
        Dataset<? extends Instance> dataset = schoolData();
        Matrix input = new JMatrix(dataset.arrayCopy());
        input.print(3, 2);
        Preferences pref = NbPreferences.forModule(HACTest.class);
        pref.put(AgglParams.LINKAGE, SingleLinkage.name);
        pref.putBoolean(AgglParams.CLUSTER_ROWS, false);
        HierarchicalResult result = subject.hierarchy(input, dataset, pref);
        Matrix similarityMatrix = result.getProximityMatrix();
        assertNotNull(similarityMatrix);
        assertEquals(similarityMatrix.rowsCount(), dataset.attributeCount());
        assertEquals(similarityMatrix.columnsCount(), dataset.attributeCount());

        result.getTreeData().print();

    }

    /**
     * Test of hierarchy method, of class HC1.
     */
    @Test
    public void testHierarchy_Matrix_Preferences() {
    }

    /**
     * Test of cluster method, of class HC1.
     */
    @Test
    public void testCluster_Dataset() {
    }

}