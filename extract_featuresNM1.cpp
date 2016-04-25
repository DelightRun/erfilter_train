#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <cstdio>
#include <cstdlib>

using namespace cv;
using namespace std;

int GroundTruth(Mat &_originalImage) {

    Mat originalImage(_originalImage.rows + 2, _originalImage.cols + 2, _originalImage.type());
    // make border with white color
    copyMakeBorder(_originalImage, originalImage, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0, 0, 0));

    // black character with white backgound
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

    // white character with black background
    threshold(originalImage, binaryImage, thresholdValue, maxValue, THRESH_BINARY);

    int regionsCount = 0;
    int totalPixelCount = binaryImage.rows * binaryImage.cols;
    Point seedPoint;
    Rect rectFilled;
    int valuesSum, q1, q2, q3; // AKA C1, C2, C3 in paper
    bool p00, p10, p01, p11;

    for (int i = 0; i < totalPixelCount; i++) {
        // find character(white)
        if (binaryImage.data[i] == maxValue) {
            seedPoint.x = i % binaryImage.cols;
            seedPoint.y = i / binaryImage.cols;

            // skip border
            if ((seedPoint.x == 0) || (seedPoint.y == 0) || (seedPoint.x == binaryImage.cols - 1) || (seedPoint.y == binaryImage.rows - 1)) {
                continue;
            }

            regionsCount++;

            // get character area(rect)
            size_t pixelsFilled = floodFill(binaryImage, seedPoint, middleScalar, &rectFilled);

            perimeter = 0;
            q1 = 0;
            q2 = 0;
            q3 = 0;

            int crossings[rectFilled.height];
            for (int j = 0; j < rectFilled.height; j++) {
                crossings[j] = 0;
            }

            // calculate features
            for (ry = rectFilled.y - 1; ry <= rectFilled.y + rectFilled.height; ry++) {
                for (rx = rectFilled.x - 1; rx <= rectFilled.x + rectFilled.width; rx++) {
                    // FEATURE: crossing
                    if ((binaryImage.at<uint8_t>(ry, rx - 1) != binaryImage.at<uint8_t>(ry, rx)) && (binaryImage.at<uint8_t>(ry, rx - 1) + binaryImage.at<uint8_t>(ry, rx) == middleValue + zeroValue)) {
                        crossings[ry - rectFilled.y]++;
                    }

                    // FEATURE: perimeter
                    if (binaryImage.at<uint8_t>(ry, rx) == middleValue) {
                        for (di = 0; di < neigborsCount; di++) {
                            int xNew = rx + dx[di];
                            int yNew = ry + dy[di];

                            if (binaryImage.at<uint8_t>(yNew, xNew) == zeroValue) {
                                perimeter++;
                            }
                        }
                    }

                    // FEATURE: euler number
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
                printf("Non-integer Euler number");
                exit(0);
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

            //Features used in the first stage classifier
            //aspect ratio (w/h), compactness (sqrt(a/p)), number of holes (1 − η), and a horizontal crossings feature (cˆ = median {c_1*w/6, c_3*w/6, c_5*w/6}) which estimates number of character strokes in horizontal projection
            if ((rectFilled.width >= 20) && (rectFilled.height >= 20)) // TODO find a better way to select good negative examples
                printf("%f,%f,%f,%f\n",
                       /* aspect ratio - width/height */ (float) rectFilled.width / rectFilled.height,
                       /* compactness - sqrt(area/perimeter) */ sqrt(pixelsFilled) / perimeter,
                       /* number of holes - (1-η)*/ (float) (1 - q1),
                       /* a horiontal crossing feature - median(c[1*w/6], c[3*w/6], c[5*w/6]) */ (float) m_crossings.at(1));

            // erase computed region with background color
            floodFill(binaryImage, seedPoint, zeroScalar);
        }
    }
}

int main(int argc, char **argv) {
    Mat originalImage;

    if (argc == 1) {
        exit(0);
    } else {
        originalImage = imread(argv[1], 0);
        // originalImage = 255 - originalImage;
    }

    GroundTruth(originalImage);

    return 0;
}
