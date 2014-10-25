package org.clueminer.clustering.benchmark;

import com.google.common.collect.ObjectArrays;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.LinkedList;
import org.clueminer.clustering.api.AgglomerativeClustering;
import org.clueminer.dataset.benchmark.GnuplotHelper;
import org.clueminer.dataset.benchmark.PointTypeIterator;
import org.clueminer.report.BigORes;
import org.clueminer.report.Reporter;
import org.openide.util.Exceptions;

/**
 *
 * @author Tomas Barton
 */
public class GnuplotReporter extends GnuplotHelper implements Reporter {

    private final String dataDir;
    private final File dataFile;
    private final String folder;
    private final AgglomerativeClustering[] algorithms;
    private final LinkedList<String> plots;

    public GnuplotReporter(String folder, String[] opts, AgglomerativeClustering[] algorithms, String suffix) {
        this.dataDir = folder + File.separatorChar + "data";
        mkdir(dataDir);
        this.dataFile = new File(dataDir + File.separatorChar + "results-" + suffix + ".csv");
        this.algorithms = algorithms;
        this.folder = folder;
        this.plots = new LinkedList<>();
        writeHeader(opts);

        String memPath = dataDir + File.separatorChar + "mem" + suffix + ".gpt";
        String cpuPath = dataDir + File.separatorChar + "cpu" + suffix + ".gpt";

        writePlotScript(new File(memPath), dataFile, 7, "memory", 2, 3);
        writePlotScript(new File(cpuPath), dataFile, 7, "CPU", 2, 4);

    }

    private void writeHeader(String[] opts) {
        String[] head = new String[]{"label", "avg time (ms)", "memory (MB)", "total time (s)", "tps", "repeats"};
        String[] line = ObjectArrays.concat(head, opts, String.class);
        writeCsvLine(dataFile, line, false);
    }

    /**
     *
     * @param result
     */
    @Override
    public void finalResult(BigORes result) {
        String[] res = new String[]{result.getLabel(), result.avgTimeMs(),
            result.totalMemoryInMb(), result.totalTimeInS(), result.tps(),
            result.measurements()
        };
        String[] line = ObjectArrays.concat(res, result.getOpts(), String.class);
        writeCsvLine(dataFile, line, true);
    }

    /**
     *
     * @param file     to write Gnuplot script
     * @param dataFile
     * @param labelPos column of label which is used for data rows in chart
     * @param type
     * @param x
     * @param y
     */
    private void writePlotScript(File file, File dataFile, int labelPos, String type, int x, int y) {
        PrintWriter template;
        try {
            template = new PrintWriter(file, "UTF-8");
            template.write(plotComplexity(labelPos, type, x, y, dataFile.getName(), algorithms));
            template.close();
        } catch (FileNotFoundException | UnsupportedEncodingException ex) {
            Exceptions.printStackTrace(ex);
        }
        plots.add(withoutExtension(file));
    }

    private String plotComplexity(int labelPos, String yLabel, int x, int y, String dataFile, AgglomerativeClustering[] algorithms) {
        String res = "set datafile separator \",\"\n"
                + "set key outside bottom horizontal box\n"
                + "set title \"foo bar\"\n"
                + "set xlabel \"data size\" font \"Times,7\"\n"
                + "set ylabel \"" + yLabel + "\" font \"Times,7\"\n"
                //   + "set xtics 0,0.5 nomirror\n"
                //   + "set ytics 0,0.5 nomirror\n"
                + "set mytics 2\n"
                + "set mx2tics 2\n"
                + "set grid\n"
                + "set pointsize 0.5\n";
        int i = 0;
        int last = algorithms.length - 1;
        PointTypeIterator pti = new PointTypeIterator();
        for (AgglomerativeClustering alg : algorithms) {
            if (i == 0) {
                res += "plot ";
            }
            res += "\"< awk -F\\\",\\\" '{if($" + labelPos + " == \\\"" + alg.getName() + "\\\") print}' " + dataFile + "\" u " + x + ":" + y + " t \"" + alg.getName() + "\" w p pt " + pti.next();
            if (i != last) {
                res += ", \\\n";
            } else {
                res += "\n";
            }

            i++;
        }
        return res;
    }

    /**
     * Should be called when all plot files are written
     */
    public void finish() {
        writePlotScript(folder);
    }

    private void writePlotScript(String dataDir) {
        try {
            bashPlotScript(plots.toArray(new String[plots.size()]), dataDir, "set term pdf font 'Times-New-Roman,8'", "pdf");
            bashPlotScript(plots.toArray(new String[plots.size()]), dataDir, "set terminal pngcairo size 1024,768 enhanced font 'Verdana,10'", "png");

        } catch (FileNotFoundException ex) {
            Exceptions.printStackTrace(ex);
        } catch (UnsupportedEncodingException ex) {
            Exceptions.printStackTrace(ex);
        } catch (IOException ex) {
            Exceptions.printStackTrace(ex);
        }
    }

}
