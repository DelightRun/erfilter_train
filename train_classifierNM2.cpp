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
    cvml->setTrainTestSplitRatio(0.8, true);

    Ptr<Boost> classifier;

    ifstream ifile("./trained_classifierNM2.xml");
    if (ifile) {
        //The file exists, so we don't want to train
        printf("Found trained_boost_char.xml file, remove it if you want to retrain with new data ... \n");
        classifier = StatModel::load<Boost>("./trained_classifierNM2.xml");
    } else {
        //Train with 100 features
        printf("Training ... \n");
        classifier = Boost::create();
        classifier->setBoostType(Boost::REAL);
        classifier->setWeakCount(100);
        classifier->setWeightTrimRate(0.0);
        classifier->train(cvml);
    }

    //Calculate the test and train errors
    Mat train_responses, test_responses;
    float fl1 = classifier->calcError(cvml, false, train_responses);
    float fl2 = classifier->calcError(cvml, true, test_responses);
    printf("Error train %f \n", fl1);
    printf("Error test %f \n", fl2);

    //Try a char
    Mat sample = (Mat_<float>(1, 7) << 0.870690, 0.096485, 2.000000, 2.000000, 0.137080, 1.269940, 2.000000);
    float prediction = classifier->predict(sample, noArray(), 0);
    float votes = classifier->predict(sample, noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT);

    printf("\n The char sample is predicted as: %f (with number of votes = %f)\n", prediction, votes);
    printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n", (float) 1 - (float) 1 / (1 + exp(-2 * votes)));

    //Try a NONchar
    //static const float arr2[] = {0,1.500000,0.072162,0.000000,8.000000,0.188095,1.578947,16.000000};
    Mat sample2 = (Mat_<float>(1, 7) << 0.565217, 0.103749, 1.000000, 2.000000, 0.032258, 1.525692, 10.000000);
    prediction = classifier->predict(sample2, noArray(), 0);
    votes = classifier->predict(sample2, noArray(), DTrees::PREDICT_SUM | StatModel::RAW_OUTPUT);

    printf("\n The non_char sample is predicted as: %f (with number of votes = %f)\n", prediction, votes);
    printf(" Class probability (using Logistic Correction) is P(r|character) = %f\n\n", (float) 1 - (float) 1 / (1 + exp(-2 * votes)));

    // Save the trained classifier
    classifier->save(string("./trained_classifierNM2.xml"));

    return EXIT_SUCCESS;
}
