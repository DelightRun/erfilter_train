#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define PI 3.14159265

using namespace cv;
using namespace std;
using namespace boost::filesystem;

struct Feature {
    float aspect_ratio;
    float compactness;
    float num_holes;
    float crossing;
    float holes_ratio;
    float convex_hull_ratio;
    float num_inflexion_points;

    Feature(float _aspect_ratio,
            float _compactness,
            float _num_holes,
            float _crossing,
            float _holes_ratio,
            float _convex_hull_ratio,
            float _num_inflexion_points) : aspect_ratio(_aspect_ratio),
                                           compactness(_compactness),
                                           num_holes(_num_holes),
                                           crossing(_crossing),
                                           holes_ratio(_holes_ratio),
                                           convex_hull_ratio(_convex_hull_ratio),
                                           num_inflexion_points(_num_inflexion_points) {}
};

vector<Feature> extract_features(Mat &_originalImage) {

    Mat originalImage(_originalImage.rows + 2, _originalImage.cols + 2, _originalImage.type());
    copyMakeBorder(_originalImage, originalImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

    Mat binaryImage(originalImage.size(), CV_8UC1);

    uchar thresholdValue = 100;
    uchar maxValue = 255;
    uchar middleValue = 192;
    uchar zeroValue = 0;
    Scalar middleScalar(middleValue);
    Scalar zeroScalar(zeroValue);

    static int neigborsCount = 4;
    static int dx[] = {-1, 0, 0, 1};
    static int dy[] = {0, -1, 1, 0};
    int di, rx, ry;
    int perimeter;

    vector<Feature> result;

    threshold(originalImage, binaryImage, thresholdValue, maxValue, THRESH_BINARY);

    int regionsCount = 0;
    int totalPixelCount = binaryImage.rows * binaryImage.cols;
    Point seedPoint;
    Rect rectFilled;
    int valuesSum, q1, q2, q3;
    bool p00, p10, p01, p11;

    for (int i = 0; i < totalPixelCount; i++) {
        if (binaryImage.data[i] == maxValue) {
            seedPoint.x = i % binaryImage.cols;
            seedPoint.y = i / binaryImage.cols;

            if ((seedPoint.x == 0) || (seedPoint.y == 0) || (seedPoint.x == binaryImage.cols - 1) || (seedPoint.y == binaryImage.rows - 1)) {
                continue;
            }

            regionsCount++;

            size_t pixelsFilled = floodFill(binaryImage, seedPoint, middleScalar, &rectFilled);

            perimeter = 0;
            q1 = 0;
            q2 = 0;
            q3 = 0;

            int crossings[rectFilled.height];
            for (int j = 0; j < rectFilled.height; j++) {
                crossings[j] = 0;
            }

            for (ry = rectFilled.y; ry < rectFilled.y + rectFilled.height; ry++) {
                for (rx = rectFilled.x; rx < rectFilled.x + rectFilled.width; rx++) {
                    if ((binaryImage.at<uint8_t>(ry, rx - 1) != binaryImage.at<uint8_t>(ry, rx)) && (binaryImage.at<uint8_t>(ry, rx - 1) + binaryImage.at<uint8_t>(ry, rx) == middleValue + zeroValue)) {
                        crossings[ry - rectFilled.y]++;
                    }

                    if (binaryImage.at<uint8_t>(ry, rx) == middleValue) {
                        for (di = 0; di < neigborsCount; di++) {
                            int xNew = rx + dx[di];
                            int yNew = ry + dy[di];

                            if (binaryImage.at<uint8_t>(yNew, xNew) == zeroValue) {
                                perimeter++;
                            }
                        }
                    }

                    p00 = binaryImage.at<uint8_t>(ry, rx) == middleValue;
                    p01 = binaryImage.at<uint8_t>(ry, rx + 1) == middleValue;
                    p10 = binaryImage.at<uint8_t>(ry + 1, rx) == middleValue;
                    p11 = binaryImage.at<uint8_t>(ry + 1, rx + 1) == middleValue;
                    valuesSum = p00 + p01 + p10 + p11;

                    if (valuesSum == 1)
                        q1++;
                    else if (valuesSum == 3)
                        q2++;
                    else if ((valuesSum == 2) && (p00 == p11))
                        q3++;
                }
            }

            q1 = q1 - q2 + 2 * q3;
            if (q1 % 4 != 0) {
                continue;
            }
            q1 /= 4;

#ifdef DEBUG
            printf("New region: %d\n", regionsCount);
            printf("Area: %d\n", (int) pixelsFilled);
            printf("Bounding box (%d; %d) + (%d; %d)\n", rectFilled.x - 1, rectFilled.y - 1, rectFilled.width, rectFilled.height);
            printf("Perimeter: %d\n", (int) perimeter);
            printf("Euler number: %d\n", q1);
            printf("Crossings: ");
            for (int j = 0; j < rectFilled.height; j++) {
                printf("%d ", crossings[j]);
            }
            printf("\n****************************\n");
#endif

            vector<int> m_crossings;
            m_crossings.push_back(crossings[(int) rectFilled.height / 6]);
            m_crossings.push_back(crossings[(int) 3 * rectFilled.height / 6]);
            m_crossings.push_back(crossings[(int) 5 * rectFilled.height / 6]);
            sort(m_crossings.begin(), m_crossings.end());

            //Features used in the second stage classifier
            if ((rectFilled.width >= 5) && (rectFilled.height >= 5)) // TODO find a better way to select good negative examples
            {
                Mat region = Mat::zeros(binaryImage.rows + 2, binaryImage.cols + 2, CV_8UC1);
                int newMaskVal = 255;
                int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE;
                Rect rect;
                floodFill(binaryImage, region, seedPoint, zeroScalar, &rect, Scalar(), Scalar(), flags);
                rect.width += 2;
                rect.height += 2;
                region = region(rect);

                vector<vector<Point> > contours;
                vector<Point> contour_poly;
                vector<Vec4i> hierarchy;
                findContours(region, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE, Point(0, 0));
                //TODO check epsilon parameter of approxPolyDP (set empirically) : we want more precission if the region is very small because otherwise we'll loose all the convexities
                approxPolyDP(Mat(contours[0]), contour_poly, (float) min(rect.width, rect.height) / 17, true);

                bool was_convex = false;
                int num_inflexion_points = 0;
                for (int p = 0; p < (int) contour_poly.size(); p++) {
                    int p_prev = p - 1;
                    int p_next = p + 1;
                    if (p_prev == -1)
                        p_prev = (int) contour_poly.size() - 1;
                    if (p_next == (int) contour_poly.size())
                        p_next = 0;

                    double angle_next = atan2((contour_poly[p_next].y - contour_poly[p].y), (contour_poly[p_next].x - contour_poly[p].x));
                    double angle_prev = atan2((contour_poly[p_prev].y - contour_poly[p].y), (contour_poly[p_prev].x - contour_poly[p].x));
                    if (angle_next < 0)
                        angle_next = 2. * PI + angle_next;

                    double angle = (angle_next - angle_prev);
                    if (angle > 2. * PI)
                        angle = angle - 2. * PI;
                    else if (angle < 0)
                        angle = 2. * PI + abs(angle);

                    if (p > 0) {
                        if (((angle > PI) && (!was_convex)) || ((angle < PI) && (was_convex)))
                            num_inflexion_points++;
                    }
                    was_convex = (angle > PI);
                }

                floodFill(region, Point(0, 0), Scalar(255), 0);
                int holes_area = region.cols * region.rows - countNonZero(region);

                vector<Point> hull;
                convexHull(contours[0], hull, false);
                int hull_area = contourArea(hull);

                result.push_back(Feature(
                    /* aspect ratio - width/height */ (float) rectFilled.width / rectFilled.height,
                    /* compactness - sqrt(area/perimeter) */ sqrt(pixelsFilled) / perimeter,
                    /* number of holes - (1-η)*/ (float) (1 - q1),
                    /* a horiontal crossing feature - median(c[1*w/6], c[3*w/6], c[5*w/6]) */ (float) m_crossings.at(1),
                    (float) holes_area / pixelsFilled,
                    (float) hull_area / contourArea(contours[0]),
                    (float) num_inflexion_points));
#ifdef DEBUG
                printf("%f,%f,%f,%f,%f,%f,%f\n",
                       /* aspect ratio - width/height */ (float) rectFilled.width / rectFilled.height,
                       /* compactness - sqrt(area/perimeter) */ sqrt(pixelsFilled) / perimeter,
                       /* number of holes - (1-η)*/ (float) (1 - q1),
                       /* a horiontal crossing feature - median(c[1*w/6], c[3*w/6], c[5*w/6]) */ (float) m_crossings.at(1),
                       (float) holes_area / pixelsFilled,
                       (float) hull_area / contourArea(contours[0]),
                       (float) num_inflexion_points);
#endif
            }

            floodFill(binaryImage, seedPoint, zeroScalar);
        }
    }

    return result;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        return 0;
    }

    path data_dir(argv[1]);
    if (!is_directory(data_dir)) {
        fprintf(stderr, "not a directory");
        return 0;
    }

    path positive_dir = data_dir / "char";
    if (!is_directory(positive_dir)) {
        fprintf(stderr, "no dir 'char'");
        return 0;
    }

    path negative_dir = data_dir / "nonchar";
    if (!is_directory(negative_dir)) {
        fprintf(stderr, "no dir 'nonchar'");
        return 0;
    }

    FILE *fnm1 = fopen("./training_data/datasetNM1.csv", "w");
    FILE *fnm2 = fopen("./training_data/datasetNM2.csv", "w");

    const directory_iterator end_iter;

    printf("Processing Positive Samples...\n");
    for (directory_iterator iter(positive_dir); iter != end_iter; iter++) {
        if (!is_regular_file(iter->status()))
            continue;

        Mat image = imread(iter->path().c_str(), 0);
        if (!image.data)
            continue;

        vector<Feature> features = extract_features(image);
        for (vector<Feature>::const_iterator feature = features.begin(); feature != features.end(); feature++) {
            fprintf(fnm1, "C,%f,%f,%f,%f\n",
                    feature->aspect_ratio,
                    feature->compactness,
                    feature->num_holes,
                    feature->crossing);
            fprintf(fnm2, "C,%f,%f,%f,%f,%f,%f,%f\n",
                    feature->aspect_ratio,
                    feature->compactness,
                    feature->num_holes,
                    feature->crossing,
                    feature->holes_ratio,
                    feature->convex_hull_ratio,
                    feature->num_inflexion_points);
        }
    }

    printf("Processing Negative Samples...\n");
    for (directory_iterator iter(negative_dir); iter != end_iter; iter++) {
        if (!is_regular_file(iter->status()))
            continue;

        Mat image = imread(iter->path().c_str(), 0);
        if (!image.data)
            continue;

        vector<Feature> features = extract_features(image);
        for (vector<Feature>::const_iterator feature = features.begin(); feature != features.end(); feature++) {
            fprintf(fnm1, "N,%f,%f,%f,%f\n",
                    feature->aspect_ratio,
                    feature->compactness,
                    feature->num_holes,
                    feature->crossing);
            fprintf(fnm2, "N,%f,%f,%f,%f,%f,%f,%f\n",
                    feature->aspect_ratio,
                    feature->compactness,
                    feature->num_holes,
                    feature->crossing,
                    feature->holes_ratio,
                    feature->convex_hull_ratio,
                    feature->num_inflexion_points);
        }
    }

    fclose(fnm1);
    fclose(fnm2);

    return 0;
}
