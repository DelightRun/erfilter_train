cmake_minimum_required(VERSION 3.5)
project(erfilter_train)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS REQUIRED filesystem system)

add_executable(extract_features extract_features.cpp)
add_executable(train_classifiers train_classifiers.cpp)

target_link_libraries(extract_features ${OpenCV_LIBS} ${Boost_LIBRARIES})
target_link_libraries(train_classifiers ${OpenCV_LIBS})
