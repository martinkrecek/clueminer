package org.clueminer.explorer.gui;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.List;
import javax.swing.DefaultComboBoxModel;
import javax.swing.JButton;
import javax.swing.JComboBox;
import javax.swing.JLabel;
import javax.swing.JSlider;
import javax.swing.JToolBar;
import javax.swing.SwingConstants;
import org.clueminer.clustering.api.ClusterEvaluation;
import org.clueminer.clustering.api.ClusteringAlgorithm;
import org.clueminer.evolution.api.EvolutionFactory;
import org.clueminer.evolution.gui.EvolutionExport;
import org.clueminer.explorer.ToolbarListener;
import org.clueminer.utils.Props;
import org.openide.DialogDescriptor;
import org.openide.DialogDisplayer;
import org.openide.NotifyDescriptor;
import org.openide.util.ImageUtilities;
import org.openide.util.NbBundle;

/**
 *
 * @author Tomas Barton
 */
public class ExplorerToolbar extends JToolBar {

    private JComboBox comboEvolution;
    private javax.swing.JSlider sliderGenerations;
    private ToolbarListener listener;
    private JButton btnSingle;
    private JButton btnStart;
    private JButton btnFunction;
    private JButton btnExport;
    private EvalFuncPanel functionPanel;
    private ExportPanel exportPanel;
    private ClusterAlgPanel algPanel;

    public ExplorerToolbar() {
        super(SwingConstants.HORIZONTAL);
        initComponents();
    }

    public void setListener(ToolbarListener listener) {
        this.listener = listener;
    }

    private void initComponents() {
        this.setFloatable(false);
        this.setRollover(true);
        btnSingle = new JButton(ImageUtilities.loadImageIcon("org/clueminer/explorer/clustering16.png", false));
        btnSingle.setToolTipText("Run single clustering");
        btnSingle.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                if (algPanel == null) {
                    algPanel = new ClusterAlgPanel();
                }
                DialogDescriptor dd = new DialogDescriptor(algPanel, NbBundle.getMessage(ExplorerToolbar.class, "AlgorithmPanel.title"));
                if (DialogDisplayer.getDefault().notify(dd).equals(NotifyDescriptor.OK_OPTION)) {
                    ClusteringAlgorithm alg = algPanel.getAlgorithm();
                    if (listener != null) {
                        Props p = algPanel.getProps();
                        listener.runClustering(alg, p);
                    }
                }
            }
        });
        add(btnSingle);

        comboEvolution = new javax.swing.JComboBox();
        comboEvolution.setModel(new DefaultComboBoxModel(initEvolution()));

        comboEvolution.addActionListener(new java.awt.event.ActionListener() {
            @Override
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                if (listener != null) {
                    listener.evolutionAlgorithmChanged(evt);
                }
            }
        });

        add(comboEvolution);
        addSeparator();
        add(new JLabel("generations:"));

        sliderGenerations = new JSlider(SwingConstants.HORIZONTAL);
        sliderGenerations.setMaximum(200);
        sliderGenerations.setMinimum(10);
        sliderGenerations.setValue(1);
        add(sliderGenerations);

        btnStart = new JButton("Start Clustering");

        btnStart.setFocusable(false);
        btnStart.setHorizontalTextPosition(javax.swing.SwingConstants.CENTER);
        btnStart.setVerticalTextPosition(javax.swing.SwingConstants.BOTTOM);
        btnStart.addActionListener(new java.awt.event.ActionListener() {
            @Override
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                if (listener != null) {
                    listener.startEvolution(evt, (String) comboEvolution.getSelectedItem());
                }

            }
        });
        add(btnStart);
        btnFunction = new JButton(ImageUtilities.loadImageIcon("org/clueminer/explorer/function16.png", false));
        btnFunction.setToolTipText("Choose evaluation function");
        btnFunction.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                if (functionPanel == null) {
                    functionPanel = new EvalFuncPanel();
                }
                DialogDescriptor dd = new DialogDescriptor(functionPanel, NbBundle.getMessage(ExplorerToolbar.class, "FunctionPanel.title"));
                if (DialogDisplayer.getDefault().notify(dd).equals(NotifyDescriptor.OK_OPTION)) {
                    ClusterEvaluation eval = functionPanel.getEvaluator();
                    if (eval != null) {
                        if (listener != null) {
                            listener.evaluatorChanged(eval);
                        }
                    }
                }
            }
        });
        add(btnFunction);
        btnExport = new JButton(ImageUtilities.loadImageIcon("org/clueminer/explorer/save16.png", false));
        btnExport.setToolTipText("Export results");
        btnExport.addActionListener(new ActionListener() {

            @Override
            public void actionPerformed(ActionEvent e) {
                if (exportPanel == null) {
                    exportPanel = new ExportPanel();
                }
                DialogDescriptor dd = new DialogDescriptor(exportPanel, NbBundle.getMessage(ExplorerToolbar.class, "ExplorerToolbar.title"));
                if (DialogDisplayer.getDefault().notify(dd).equals(NotifyDescriptor.OK_OPTION)) {
                    EvolutionExport exp = exportPanel.getExporter();
                    if (exp != null) {
                        if (listener != null) {
                            exp.setEvolution(listener.currentEvolution());
                            exp.export();
                        }
                    }
                }
            }
        });
        add(btnExport);
        addSeparator();
    }

    private String[] initEvolution() {
        EvolutionFactory ef = EvolutionFactory.getInstance();
        List<String> list = ef.getProviders();
        String[] res = new String[list.size()];
        int i = 0;
        for (String s : list) {
            res[i++] = s;
        }
        return res;
    }

    public void evolutionStarted() {
        btnStart.setEnabled(false);
    }

    public void evolutionFinished() {
        btnStart.setEnabled(true);
    }

    public int getGenerations() {
        return sliderGenerations.getValue();
    }

}
