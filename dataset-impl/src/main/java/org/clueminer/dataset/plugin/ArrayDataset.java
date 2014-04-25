package org.clueminer.dataset.plugin;

import java.util.*;
import java.util.Map.Entry;
import javax.swing.JComponent;
import org.clueminer.attributes.AttributeFactoryImpl;
import org.clueminer.dataset.api.Attribute;
import org.clueminer.dataset.api.AttributeBuilder;
import org.clueminer.dataset.api.Dataset;
import org.clueminer.dataset.api.Instance;
import org.clueminer.dataset.api.InstanceBuilder;
import org.math.plot.Plot2DPanel;

/**
 * Dataset with fixed number of items
 *
 * @param <E>
 *
 * - contains(Object o) ~ O(n) - can allocate Instance at any position
 *
 *
 * @author Tomas Barton
 */
public class ArrayDataset<E extends Instance> extends AbstractArrayDataset<E> implements Dataset<E> {

    private static final long serialVersionUID = -5482153886671625555L;
    private Instance[] data;
    private InstanceBuilder builder;
    private AttributeBuilder attributeBuilder;
    private final TreeSet<Object> classes = new TreeSet<Object>();
    protected Attribute[] attributes;
    private int attrCnt = 0;
    /**
     * (n - 1) is index of last inserted item, n itself represents current
     * number of instances in this dataset
     */
    private int n = 0;

    public ArrayDataset(int instancesCapacity, int attributesCnt) {
        data = new Instance[instancesCapacity];
        attributes = new Attribute[attributesCnt];
    }

    @Override
    public SortedSet<Object> getClasses() {
        return classes;
    }

    @Override
    public boolean add(Instance inst) {
        ensureCapacity(n);

        data[n] = inst;
        inst.setIndex(n);
        n++;
        return true;
    }


    /*
     * public boolean addAll(Collection<? extends Instance> c) { throw new
     * UnsupportedOperationException("Not supported yet."); }
     */
    @Override
    public boolean addAll(Dataset<E> d) {
        Iterator<E> it = d.iterator();
        while (n < data.length && it.hasNext()) {
            Instance i = it.next();
            data[n++] = i;
        }
        return !it.hasNext();
    }

    @Override
    public E instance(int index) {
        if (hasIndex(index)) {
            return get(index);
        } else if (index == size()) {
            E inst = (E) builder().create(this.attributeCount());
            add(inst);
            return inst;
        }
        throw new ArrayIndexOutOfBoundsException("can't get instance at position: " + index);
    }

    /**
     * We have to check data.length because we can have empty instances:
     *
     * [ null, null, null, Instance(3) ] -- size() == 1
     *
     * @param idx
     * @return
     */
    @Override
    public boolean hasIndex(int idx) {
        return idx >= 0 && idx < data.length && data[idx] != null;
    }

    @Override
    public E getRandom(Random rand) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public int size() {
        return n;
    }

    /**
     *
     * @return true when dataset is empty, false otherwise
     */
    @Override
    public boolean isEmpty() {
        return (size() == 0);
    }

    @Override
    public int classIndex(Object clazz) {
        if (clazz != null) {
            return this.getClasses().headSet(clazz).size();
        } else {
            return -1;
        }

    }

    @Override
    public Object classValue(int index) {
        int i = 0;
        for (Object o : this.classes) {
            if (i == index) {
                return o;
            }
            i++;
        }
        return null;
    }

    @Override
    public InstanceBuilder builder() {
        if (builder == null) {
            builder = new DoubleArrayFactory('.');
        }
        return builder;
    }

    @Override
    public AttributeBuilder attributeBuilder() {
        if (attributeBuilder == null) {
            attributeBuilder = new AttributeFactoryImpl();
        }
        return attributeBuilder;
    }

    /**
     * Real attribute count, doesn't include null attributes
     *
     * @return actual number of attributes
     */
    @Override
    public int attributeCount() {
        return attrCnt;
    }

    /**
     * Get i-th attribute instance
     *
     * @param i
     * @return
     */
    @Override
    public Attribute getAttribute(int i) {
        return attributes[i];
    }

    /**
     * Get i-th attribute by its name
     *
     * @param attributeName
     * @return
     */
    @Override
    public Attribute getAttribute(String attributeName) {
        for (Attribute attribute : attributes) {
            if (attribute.getName().equals(attributeName)) {
                return attribute;
            }
        }
        throw new RuntimeException("Attribute with name " + attributeName + " was not found");
    }

    /**
     * Set i-th attribute (column)
     *
     * @param i
     * @param attr
     */
    @Override
    public void setAttribute(int i, Attribute attr) {
        attr.setIndex(i);
        ensureAttrSize(i);
        if (attributes[i] == null) {
            attrCnt++;
        }
        attributes[i] = attr;
    }

    public final void ensureAttrSize(int reqAttrSize) {
        if (reqAttrSize >= attributes.length) {
            int capacity = (int) (reqAttrSize * 1.618); //golden ratio :)
            if (capacity == attributeCount()) {
                capacity = 3 * reqAttrSize; // for small numbers due to int rounding we wouldn't increase the size
            }
            Attribute[] tmp = new Attribute[capacity];
            System.arraycopy(attributes, 0, tmp, 0, attrCnt);
            attributes = tmp;
        }
    }

    /**
     * Array might have free allocated space for new attributes, so copy just
     * {attrCnt} attributes
     *
     * @return reference to attribute map
     */
    @Override
    public Map<Integer, Attribute> getAttributes() {
        Map<Integer, Attribute> res = new HashMap<Integer, Attribute>();
        for (int i = 0; i < attrCnt; i++) {
            res.put(i, attributes[i]);
        }
        return res;
    }

    @Override
    public Attribute[] copyAttributes() {
        Attribute[] copy = new Attribute[attributeCount()];
        for (int i = 0; i < copy.length; i++) {
            copy[i] = (Attribute) getAttribute(i).clone();
        }
        return attributes.clone();
    }

    /**
     * Deep copy of dataset
     *
     * @return
     */
    @Override
    public Dataset<E> copy() {
        SampleDataset out = new SampleDataset();
        Instance inst;
        for (int i = 0; i < size(); i++) {
            inst = instance(i);
            out.add(inst.copy());
        }
        return out;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("ArrayDataset [ size= " + size() + " \n");
        for (int i = 0; i < size(); i++) {
            str.append(classValue(i)).append(">> ").append(this.instance(i).toString());
        }
        str.append("\n ]");
        return str.toString();
    }

    @Override
    public double[][] arrayCopy() {
        double[][] res = new double[this.size()][this.attributeCount()];
        int i = 0;
        int cols = this.attributeCount();
        if (cols <= 0) {
            throw new ArrayIndexOutOfBoundsException("given dataset has width " + cols);
        }
        for (Instance inst : this) {
            for (int j = 0; j < inst.size(); j++) {
                res[i][j] = inst.value(j);///scaleToRange((float)inst.value(j), min, max, -10, 10);
            }
            i++;
        }
        return res;
    }

    @Override
    public double getAttributeValue(String attributeName, int instanceIdx) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public double getAttributeValue(Attribute attribute, int instanceIdx) {
        return data[instanceIdx].value(attribute.getIndex());
    }

    @Override
    public double getAttributeValue(int attributeIndex, int instanceIdx) {
        return data[instanceIdx].value(attributeIndex);
    }

    @Override
    public void setAttributes(Map<Integer, Attribute> attrs) {
        ensureAttrSize(attrs.size());

        for (Entry<Integer, Attribute> entry : attrs.entrySet()) {
            this.setAttribute(entry.getKey(), entry.getValue());
        }
    }

    @Override
    public void setAttributeValue(String attributeName, int instanceIdx, double value) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    @Override
    public JComponent getPlotter() {
        Plot2DPanel plot = new Plot2DPanel();

        double[] x = new double[this.size()];
        double[] y = new double[this.size()];
        // Dump.printMatrix(data.length,data[0].length,data,2,5);
        int k = 5;
        for (int j = 0; j < this.size(); j++) {
            x[j] = getAttributeValue(k, j);
        }

        k = 0;
        for (int j = 0; j < this.size(); j++) {
            //Attribute ta =  dataset.getAttribute(j);
            y[j] = getAttributeValue(k, j);

        }
        plot.addScatterPlot(getName(), x, y);
        return plot;
    }

    /**
     * Copies attributes but not data itself
     *
     * @return copy of dataset structure
     */
    @Override
    public Dataset<E> duplicate() {
        ArrayDataset<E> copy = new ArrayDataset<E>(this.size(), this.attributeCount());
        copy.attributes = this.attributes;
        copy.attrCnt = this.attrCnt;
        return copy;
    }

    @Override
    public int getCapacity() {
        if (data != null) {
            return data.length;
        }
        return 0;
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public E get(int index) {
        return (E) data[index];
    }

    @Override
    public void ensureCapacity(int req) {
        if (req >= getCapacity()) {
            int capacity = (int) (n * 1.618); //golden ratio :)
            if (capacity <= req) {
                capacity = req + 1; // for small numbers due to int rounding we wouldn't increase the size
            }

            Instance[] tmp = new Instance[capacity];
            System.arraycopy(data, 0, tmp, 0, n);
            data = tmp;
        }
    }

    /**
     * Returns <tt>true</tt> if this list contains the specified element. More
     * formally, returns <tt>true</tt> if and only if this list contains at
     * least one element <tt>e</tt> such that
     * <tt>(o==null&nbsp;?&nbsp;e==null&nbsp;:&nbsp;o.equals(e))</tt>.
     *
     * @param o element whose presence in this list is to be tested
     * @return <tt>true</tt> if this list contains the specified element
     */
    @Override
    public boolean contains(Object o) {
        return indexOf(o) >= 0;
    }

    /**
     * Returns the index of the first occurrence of the specified element in
     * this list, or -1 if this list does not contain the element. More
     * formally, returns the lowest index <tt>i</tt> such that
     * <tt>(o==null&nbsp;?&nbsp;get(i)==null&nbsp;:&nbsp;o.equals(get(i)))</tt>,
     * or -1 if there is no such index.
     *
     * @param o
     * @return
     */
    public int indexOf(Object o) {
        if (o == null) {
            for (int i = 0; i < size(); i++) {
                if (data[i] == null) {
                    return i;
                }
            }
        } else {
            for (int i = 0; i < size(); i++) {
                if (o.equals(data[i])) {
                    return i;
                }
            }
        }
        return -1;
    }

    @Override
    public Object[] toArray() {
        return data;
    }

    @Override
    public <T> T[] toArray(T[] a) {
        return (T[]) data;
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    class ArrayDatasetIterator implements Iterator<Instance> {

        private int index = 0;

        @Override
        public boolean hasNext() {
            return index < size();
        }

        @Override
        public Instance next() {
            index++;
            return instance(index - 1);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Cannot remove from dataset using the iterator.");

        }
    }

    @Override
    public Iterator<E> iterator() {
        return (Iterator<E>) new ArrayDatasetIterator();
    }
}
