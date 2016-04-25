#include <cstdlib>
#include <fstream>
#include <vector>

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;
using namespace cv;
using namespace cv::ml;

int main(int argc, char **argv) {

    //Read the data from csv file
    Ptr<TrainData> cvml = TrainData::loadFromCSV(string("char_datasetNM2.csv"), 0, 0);

    //Select 80% for the training
    cvml->setTrainTestSplitRatio(0.9, true);

    Ptr<RTrees> classifier;

      //Train with 100 features
      printf("Training ... \n");
      classifier = RTrees::create();
      classifier->train(cvml);

    //Calculate the test and train errors
    Mat train_responses, test_responses;
    float fl1 = classifier->calcError(cvml, false, train_responses);
    float fl2 = classifier->calcError(cvml, true, test_responses);
    printf("Error train %f \n", fl1);
    printf("Error test %f \n", fl2);

    // Save the trained classifier
    classifier->save(string("./trained_classifierNM2.xml"));

    return EXIT_SUCCESS;
}
