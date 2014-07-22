package org.clueminer.dendrogram;

import java.awt.Component;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import javax.swing.JFrame;
import javax.swing.SwingUtilities;
import org.clueminer.clustering.api.dendrogram.DendroViewer;
import org.clueminer.dgram.DgViewer;

/**
 * Should serve for testing DendroView component
 *
 * @author deric
 */
public class DendroView extends JFrame {

    private DendroViewer frame;
    private DendroToolbar toolbar;

    public DendroView() {
        setLayout(new GridBagLayout());

        frame = new DgViewer();
        GridBagConstraints c = new GridBagConstraints();
        c.anchor = GridBagConstraints.NORTHWEST;
        c.fill = GridBagConstraints.BOTH;
        c.gridx = 0;
        c.gridy = 1;
        c.gridwidth = 1;
        c.gridheight = 1;
        c.weightx = 1.0;
        c.weighty = 1.0;
        c.insets = new Insets(0, 0, 0, 0);
        add((Component) frame, c);

        toolbar = new DendroToolbar();
        c.gridy = 0;
        c.fill = GridBagConstraints.HORIZONTAL;
        c.weighty = 0;
        add(toolbar, c);
    }

    // this function will be run from the EDT
    private static void createAndShowGUI() throws Exception {
        DendroView hmf = new DendroView();
        hmf.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        hmf.setSize(500, 500);
        hmf.setVisible(true);
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(new Runnable() {

            @Override
            public void run() {
                try {
                    createAndShowGUI();
                } catch (Exception e) {
                    System.err.println(e);
                    e.printStackTrace();
                }
            }
        });
    }

}
