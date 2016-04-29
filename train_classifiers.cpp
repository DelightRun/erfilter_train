#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

/* AdaBoost */
void train_classifier(string datafile, string savefile) {
    //Read the data from csv file
    Ptr<TrainData> cvml = TrainData::loadFromCSV(datafile, 0, 0);
    //Select 90% for the training
    cvml->setTrainTestSplitRatio(0.9, true);

    printf("Training ... \n");
    Ptr<Boost> classifier;
    classifier = Boost::create();
    classifier->setBoostType(Boost::REAL);
    classifier->setWeakCount(100);
    classifier->setWeightTrimRate(0.0);
    classifier->setMaxDepth(1);
    classifier->train(cvml);

    //Calculate the test and train errors
    Mat train_responses, test_responses;
    float fl1 = classifier->calcError(cvml, false, train_responses);
    float fl2 = classifier->calcError(cvml, true, test_responses);
    printf("Error train %f \n", fl1);
    printf("Error test %f \n", fl2);

    // Save the trained classifier
    classifier->save(savefile);
}

int main(int argc, char* argv[]) {
    printf("Traing Classifier For Stage 1...\n");
    train_classifier(string("./training_data/datasetNM1.csv"),
                     string("./trained_classifiers/trained_classifierNM1.xml"));
    printf("Traing Classifier For Stage 2...\n");
    train_classifier(string("./training_data/datasetNM2.csv"),
                     string("./trained_classifiers/trained_classifierNM2.xml"));
}
