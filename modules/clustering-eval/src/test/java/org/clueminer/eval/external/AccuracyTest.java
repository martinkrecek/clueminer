package org.clueminer.eval.external;

import org.clueminer.fixtures.clustering.FakeClustering;
import static org.junit.Assert.assertEquals;
import org.junit.Test;

/**
 *
 * @author tombart
 */
public class AccuracyTest extends ExternalTest {

    public AccuracyTest() {
        subject = new Accuracy();
    }

    /**
     * Test of score method, of class Accuracy.
     */
    @Test
    public void testScore_Clustering_Clustering() {
        double score;
        //each cluster should have this scores:
        //Cabernet = 0.7407
        //Syrah = 0.7037
        //Pinot = 0.8889
        score = measure(FakeClustering.wineClustering(), FakeClustering.wineCorrect(), 0.6438746438746439);

        //when using class labels result should be the same
        measure(FakeClustering.wineClustering(), score);
    }

    @Test
    public void testOneClassPerCluster() {
        assertEquals(0.0, subject.score(oneClassPerCluster()), delta);
    }

    /**
     * Test of isBetter method, of class Accuracy.
     */
    @Test
    public void testCompareScore() {
        double scoreBetter = subject.score(FakeClustering.iris());
        double scoreWorser = subject.score(FakeClustering.irisWrong5());

        assertEquals(true, subject.isBetter(scoreBetter, scoreWorser));
    }

    @Test
    public void testMostlyWrong() {
        double score = subject.score(FakeClustering.irisMostlyWrong());
        System.out.println("accuracy (mw) = " + score);
        assertEquals(true, score < 0.4);
    }
}
