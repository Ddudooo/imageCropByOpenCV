package opencv;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.apache.tika.Tika;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class App {

    private static final Logger log = LogManager.getLogger(App.class);

    public static void main(String[] args) {
        log.info("Processing...");

        nu.pattern.OpenCV.loadShared();
        File imgDir = new File(
            Objects.requireNonNull(App.class.getClassLoader().getResource("images")).getFile());
        List<File> files = Arrays.asList(imgDir.listFiles());

        File outputDir = generateOutputDirectory();
        log.info("OUTPUT DIR - [{}]", outputDir.getAbsolutePath());
        files.parallelStream()
            .forEach(App::detectFace);
    }

    private static void detectFace(File img) {
        try {
            Tika tika = new Tika();
            String mediaType = tika.detect(img);
            boolean isImage = StringUtils.contains(mediaType, "image");
            if (isImage) {
                String imgName = FilenameUtils.getName(img.getAbsolutePath());
                String imgExtension = FilenameUtils.getExtension(img.getAbsolutePath());
                String imgFileSize = FileUtils.byteCountToDisplaySize(FileUtils.sizeOf((img)));
                log.info("Image[{}.{}] - {}", imgName, imgExtension, imgFileSize);
                Mat src = Imgcodecs.imread(img.getAbsolutePath());
                File xml = new File(Objects.requireNonNull(
                    App.class.getClassLoader().getResource("opencv/xml/lbpcascade_frontalface.xml"))
                    .getFile());
                CascadeClassifier classifier = new CascadeClassifier(xml.getAbsolutePath());

                MatOfRect faceDetections = new MatOfRect();
                classifier.detectMultiScale(src, faceDetections);
                log.info("Detected {} faces", faceDetections.toArray().length);
                Double padding = 50.0;

                for (Rect rect : faceDetections.toArray()) {
                    Rect cropWithPaddingRect =
                        new Rect(
                            new Point(rect.x - padding, rect.y - padding),
                            new Point(rect.x + rect.width + padding, rect.y + rect.height + padding)
                        );
                    Mat imgCrop = new Mat(src, cropWithPaddingRect);

                    String outputCropName = String
                        .format("%s/%s_CROP.%s", getOutputDirectory(), imgName, imgExtension);
                    Imgcodecs.imwrite(outputCropName, imgCrop);

                    Imgproc.rectangle(
                        src,
                        new Point(rect.x - padding, rect.y - padding),
                        new Point(rect.x + rect.width + padding, rect.y + rect.height + padding),
                        new Scalar(0, 255, 0),
                        1
                    );
                }

                String detectedImgName = String
                    .format("%s/%s_DETECTED.%s", getOutputDirectory(), imgName, imgExtension);
                Imgcodecs.imwrite(detectedImgName, src);
                log.info("Image Processed");
            }
        } catch (IOException e) {
            log.error(e.getMessage(), e);
        }
    }

    private static File generateOutputDirectory() {
        File outputDir = new File("output");
        if (outputDir.exists()) {
            try {
                log.warn("CLEAR OUTPUT DIR");
                FileUtils.forceDelete(outputDir);
            } catch (IOException e) {
                log.error(e.getMessage(), e);
            }
        }
        outputDir.mkdir();
        return outputDir;
    }

    private static String getOutputDirectory() {
        File outputDir = new File("output");
        if (!outputDir.exists()) {
            outputDir.mkdir();
        }
        return outputDir.getAbsolutePath();
    }
}
