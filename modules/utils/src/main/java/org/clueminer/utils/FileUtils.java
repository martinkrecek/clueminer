package org.clueminer.utils;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.channels.FileChannel;
import org.openide.filesystems.FileObject;
import org.openide.filesystems.FileUtil;
import org.openide.util.NbBundle;

/**
 *
 * @author Tomas Barton
 */
public final class FileUtils {

    public static String LocalFolder() {
        String result = System.getProperty("user.home") + File.separator
                + NbBundle.getMessage(
                        FileUtils.class,
                        "FOLDER_Home");
        ensureFolder(result);
        return result;
    }

    public static String LogFolder() {
        String result = LocalFolder() + File.separator + "log";
        ensureFolder(result);
        return result;
    }

    public static String LogFile() {
        String result = LogFolder() + File.separator + "log.txt";
        createFile(result);
        return result;
    }

    public static String ErrorFile() {
        String result = LogFolder() + File.separator + "err.txt";
        createFile(result);
        return result;
    }

    public static String SettingsFolder() {
        String result = LocalFolder() + File.separator + "settings";
        ensureFolder(result);
        return result;
    }

    public static String UserFile() {
        String result = SettingsFolder() + File.separator + "user.xml";
        return result;
    }

    public static String RegisterFile() {
        String result = SettingsFolder() + File.separator + "registred.xml";
        return result;
    }

    public static String cacheFolder() {
        String result = LocalFolder() + File.separator + "cache";
        ensureFolder(result);
        return result;
    }

    public static String cacheFile(String file) throws IOException {
        String filePath = cacheFolder() + File.separator + file;
        FileObject object = FileUtil.createData(new File(filePath));
        return object.getPath();
    }

    public static FileObject cacheFileObject(String fileName) throws IOException {
        return FileUtil.createData(new File(cacheFile(fileName)));
    }

    public static String getHistoryFolder() {
        String result = LocalFolder() + File.separator + "history";
        ensureFolder(result);
        return result;
    }

    public static File samplesFile() {
        String result = LocalFolder() + File.separator + "samples.xml";
        return new File(result);
    }

    public static String templatesFolder() {
        String result = LocalFolder() + File.separator + "templates";
        ensureFolder(result);
        return result;
    }

    public static File templatesFile(String fileName) {
        String result = templatesFolder() + File.separator + fileName;
        return new File(result);
    }

    public static boolean fileExists(String path) {
        File file = new File(path);
        return file.exists();
    }

    public static void removeFile(String path) {
        File file = new File(path);
        if (file.exists()) {
            try {
                FileObject fo = FileUtil.toFileObject(file);
                fo.delete();
            } catch (IOException ex) {
                ex.printStackTrace();
            }
        }
    }

    public static void createFile(String path) {
        File f = new File(path);
        try {
            FileObject file = FileUtil.createData(f);
        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }

    public static void ensureFolder(String path) {
        File dir = new File(path);
        if (!dir.exists()) {
            try {
                FileUtil.createFolder(dir);
            } catch (IOException ex) {
                ex.printStackTrace();
                throw new RuntimeException("failed to create " + dir);
            }
        }
    }

    public static void copyFile(String source, String destination) throws IOException {
        File sourceFile = new File(source);
        File destinationFile = new File(destination);
        copyFile(sourceFile, destinationFile);
    }

    public static void copyFile(File source, File destination) throws IOException {
        FileChannel in = null;
        FileChannel out = null;
        try {
            in = new FileInputStream(source).getChannel();
            out = new FileOutputStream(destination).getChannel();
            in.transferTo(0, in.size(), out);
        } finally {
            if (in != null) {
                in.close();
            }
            if (out != null) {
                out.close();
            }
        }
    }

    public static String getFileName(String folder, final String path) {
        File dir = new File(folder);
        String[] list = dir.list(new FilenameFilter() {

            @Override
            public boolean accept(File dir, String name) {
                return name.contains(path);
            }
        });

        if (list.length == 0) {
            return folder + File.separator + path;
        } else {
            return folder + File.separator + path + "(" + list.length + ")";
        }
    }
}
