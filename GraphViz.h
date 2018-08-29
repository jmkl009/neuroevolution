//
// Created by WangJingjin on 2018/7/31.
//

#ifndef OPENCV_PRACTICE_GRAPHVIZ_H
#define OPENCV_PRACTICE_GRAPHVIZ_H

//
// Created by WangJingjin on 2018/7/22.
//

#include <opencv2/opencv.hpp>
// #include <cmath>
#include "Individual.h"

using namespace cv;

inline const char* format_bool(bool b) {
    return b ? "true" : "false";
}

inline float euclideanDist(Point& p, Point& q) {
    Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

inline float clamp(float f, float lower, float upper) {
    return f < lower ? lower : (f > upper ? upper : f);
}

unsigned show_activation(Mat& image, const NeuralNetwork* network, std::vector<Point>& neuron_centers, const int id, const unsigned neuron_radius) {
    float a = network->getNeuronByIdx(id)->activation * 0.5;
    float color_comp = clamp(0.3 + 0.7 * a, 0.0, 1.0);
    Scalar color = (a < 0 ? Scalar(0.3, 0.3, color_comp) : Scalar(color_comp, color_comp, color_comp)) * 255.0;
    unsigned activation_radius = clamp(fabsf(a), 0.15 , 3.0) * neuron_radius;

    circle(image, neuron_centers[id], activation_radius, color, -1);
    return activation_radius;
}

class GraphViz {

#define INPUT_COLOR  Scalar(0, 255, 0)
#define HIDDEN_COLOR Scalar(127, 127, 127)
#define OUTPUT_COLOR Scalar(225, 225, 0)
#define BLACK        Scalar(0, 0, 0)

private:
    void arc(Mat& image, Point& in, Point& out, unsigned second_axis_length, Scalar& color, int thickness, bool up) {
        unsigned first_axis_length = (unsigned)(euclideanDist(in, out) * 0.5);

        int dif_x = abs(in.x - out.x);
        int dif_y = abs(in.y - out.y);

        double angle;
        if (dif_x == 0) {
            angle = 90;
        } else if (dif_y == 0) {
            angle = 0;
        } else {
            angle = (in.y > out.y ? atan(((float)dif_y) / dif_x) : -atan(((float)dif_y) / dif_x)) * 180 / M_PI;
        }

        if (up) {
            ellipse(image, (in + out) * 0.5, Size(first_axis_length, second_axis_length), angle, 180, 360, color, thickness);
        } else {
            ellipse(image, (in + out) * 0.5, Size(first_axis_length, second_axis_length), angle, 0, 180, color, thickness);
        }
    };

public:
    const std::vector<ConnectionGene>& genome;
    NeuralNetwork* network;
    std::vector<std::vector<NeuralNetwork::Neuron*> > neurons_by_depth;
    std::vector<Point> neuron_centers;
//    Mat img;

    const unsigned inputIdUpperBound;
    const unsigned outputIdUpperBound;

    unsigned max_width;
    unsigned max_height;
    unsigned default_neuron_num_per_layer;
    int max_line_thickness;
    unsigned max_depth;


    unsigned major_axis_length;
    unsigned minor_axis_length;
    unsigned layer_thinkness;
    unsigned layer_start;
    unsigned neuron_radius;
    float max_weight_reciprocal;

    //   std::vector<unsigned> prev_activation_radii;

    bool horizontal;
    bool showID;

    GraphViz(Individual* ind, bool showID = false) : GraphViz(ind->getConnGenome(), ind->getPhenotype(), showID) {

    }

    GraphViz(const std::vector<ConnectionGene>& genotype, NeuralNetwork* net, bool showID = false)
            : genome(genotype), network(net),
              neurons_by_depth(net->getNeuronsByDepth()),
              neuron_centers(network->getNetworkSize()),
              inputIdUpperBound(net->inputIdUpperBound), outputIdUpperBound(net->outputIdUpperBound),
              max_width(300), max_height(150), default_neuron_num_per_layer(6), max_line_thickness(3), horizontal(true),
              showID(showID) {

        max_depth = neurons_by_depth.size();

        float max_weight = 1.0; //Store this value for future use
        for (ConnectionGene const &g : genome) {
            float weight = fabsf(g.weight);
            if (max_weight < weight) {
                max_weight = weight;
            }
        }

        max_weight_reciprocal = 1.0 / max_weight;
    };

    void set() {
        major_axis_length = horizontal ? max_width : max_height;
        minor_axis_length = horizontal ? max_height : max_width;

        layer_thinkness = major_axis_length / max_depth;
        layer_start = layer_thinkness >> 1;

        unsigned max_layer_size = neurons_by_depth[0].size();
        for (unsigned i = 1; i < max_depth; i++) {
            unsigned curr_layer_size = neurons_by_depth[i].size();
            if (max_layer_size < curr_layer_size) {
                max_layer_size = curr_layer_size;
            }
        }

        const unsigned neuron_num_per_layer_divider = max_layer_size < default_neuron_num_per_layer ?
                                                      default_neuron_num_per_layer : max_layer_size;
        neuron_radius = minor_axis_length / ((neuron_num_per_layer_divider * 3) + 1); // * 3 + 1 for some space between neurons
//        unsigned neuron_center_separation = minor_axis_length / (neuron_num_per_layer_divider);


        if (horizontal) {
            unsigned layer_placement_pointer = max_width - layer_start;
            for (std::vector<NeuralNetwork::Neuron*> &layer : neurons_by_depth) {
                const unsigned neuron_center_separation = minor_axis_length / layer.size();
                unsigned neuron_placement_pointer = neuron_center_separation >> 1;
                for (NeuralNetwork::Neuron*& n : layer) {
                    neuron_centers[network->getIdxById(n->id)] = Point(layer_placement_pointer, neuron_placement_pointer);
                    neuron_placement_pointer += neuron_center_separation;
                }
                layer_placement_pointer -= layer_thinkness;
            }
        } else {
            unsigned layer_placement_pointer = layer_start;
            for (std::vector<NeuralNetwork::Neuron*> &layer : neurons_by_depth) {
                const unsigned neuron_center_separation = minor_axis_length / layer.size();
                unsigned neuron_placement_pointer = neuron_center_separation >> 1;
                for (NeuralNetwork::Neuron*& n : layer) {
                    neuron_centers[network->getIdxById(n->id)] = Point(neuron_placement_pointer, layer_placement_pointer);
                    neuron_placement_pointer += neuron_center_separation;
                }
                layer_placement_pointer += layer_thinkness;
            }
        }
    }

    void graph(int wait_miliseconds) {
        Mat image = Mat::zeros(max_height, max_width, CV_8UC3);//(max_height, max_width, CV_8UC3);

        for (ConnectionGene const &g : genome) {
            if (g.enabled) {
                float weight_ratio = fabsf(g.weight) * max_weight_reciprocal;
                int thickness = weight_ratio * (max_line_thickness - 1) + 1;
                if (!g.is_recurrent) {
                    Scalar color = g.weight < 0 ? Scalar(0, 0, 255) : Scalar(255, 0, 0);
                    line(image, neuron_centers[network->getIdxById(g.in)], neuron_centers[network->getIdxById(g.out)], color, thickness);
                } else {
                   // float color_comp = g.weight * max_weight_reciprocal * 255;
                    Scalar color = g.weight < 0 ? Scalar(0, 255, 0) : Scalar(255, 255, 255);

                    if (g.in != g.out) {
                        Point& in = neuron_centers[network->getIdxById(g.in)];
                        Point& out = neuron_centers[network->getIdxById(g.out)];

                        if (network->testBackwardRecurrency(g.in, g.out)) {
                            arc(image, in, out, neuron_radius << 1, color, thickness, false);
                        } else {
                            arc(image, out, in, neuron_radius << 1, color, thickness, true);
                        }
                    } else {

                        Point center = neuron_centers[network->getIdxById(g.in)];
                        if (horizontal) {
                            center.y += neuron_radius;
                        } else {
                            center.x += neuron_radius;
                        }
                        circle(image, center, neuron_radius, color, thickness);
                    }
                }
            }
        }

        for (unsigned i = 0; i < inputIdUpperBound; i++) {
            show_activation(image, network, neuron_centers, i, neuron_radius);
            circle(image, neuron_centers[i], neuron_radius, INPUT_COLOR, 2);
            if (showID) {
                putText(image, std::to_string(network->getNeuronByIdx(i)->id), neuron_centers[i], FONT_HERSHEY_SIMPLEX, 0.8, INPUT_COLOR, 2,
                        LINE_AA);
            }
        }

        for (unsigned i = inputIdUpperBound; i < outputIdUpperBound; i++) {
            show_activation(image, network, neuron_centers, i, neuron_radius);
            circle(image, neuron_centers[i], neuron_radius, OUTPUT_COLOR, 2);
            if (showID) {
                putText(image, std::to_string(network->getNeuronByIdx(i)->id), neuron_centers[i], FONT_HERSHEY_SIMPLEX, 0.8, OUTPUT_COLOR, 2,
                        LINE_AA);
            }
        }

        unsigned network_size = network->getNetworkSize();
        for (unsigned i = outputIdUpperBound; i < network_size; i++) {
            show_activation(image, network, neuron_centers, i, neuron_radius);
            circle(image, neuron_centers[i], neuron_radius, HIDDEN_COLOR, 2);
            if (showID) {
                putText(image, std::to_string(network->getNeuronByIdx(i)->id), neuron_centers[i], FONT_HERSHEY_SIMPLEX, 0.8, HIDDEN_COLOR, 2,
                        LINE_AA);
            }
        }

        imshow("neural network", image);
        waitKey(wait_miliseconds);
    }

//    void allocateMat() {
//        prev_activation_radii = std::vector<unsigned>(network->getNetworkSize());
//        img = Mat(max_height, max_width, CV_8UC3);
//        for (ConnectionGene const &g : genome) {
//            if (g.enabled) {
//                float weight_ratio = fabsf(g.weight) * max_weight_reciprocal;
//                float thickness = weight_ratio * (max_line_thickness - 1) + 1;
//                if (!network->testBackwardRecurrency(g.in, g.out)) {
//                    Scalar color = g.weight < 0 ? Scalar(0, 0, weight_ratio * 255) : Scalar(weight_ratio * 255, 0, 0);
//                    line(img, neuron_centers[network->getIdxById(g.in)], neuron_centers[network->getIdxById(g.out)], color, thickness);
//                } else {
//                    float color_comp = g.weight * max_weight_reciprocal * 255;
//                    Scalar color = g.weight < 0 ? Scalar(0, -color_comp, 0) : Scalar(color_comp, color_comp, color_comp);
//
//                    if (g.in != g.out) {
//                        Point& in = neuron_centers[network->getIdxById(g.in)];
//                        Point& out = neuron_centers[network->getIdxById(g.out)];
//                        ellipse(img, (in + out) * 0.5, Size(euclideanDist(in, out) * 0.5, neuron_radius << 1), horizontal ? 0 : 90, 0, 180, color);
//                    } else {
//
//                        Point center = neuron_centers[network->getIdxById(g.in)];
//                        if (horizontal) {
//                            center.y += neuron_radius;
//                        } else {
//                            center.x += neuron_radius;
//                        }
//                        circle(img, center, neuron_radius, color, 1);
//                    }
//                }
//            }
//        }
//
//        for (unsigned i = 0; i < inputIdUpperBound; i++) {
//            prev_activation_radii[i] = show_activation(img, network, neuron_centers, i, neuron_radius);
//            circle(img,  neuron_centers[i], neuron_radius, INPUT_COLOR, 2);
//        }
//
//        for (unsigned i = inputIdUpperBound; i < outputIdUpperBound; i++) {
//            prev_activation_radii[i] = show_activation(img, network, neuron_centers, i, neuron_radius);
//            circle(img, neuron_centers[i], neuron_radius, OUTPUT_COLOR, 2);
//        }
//
//        unsigned network_size = network->getNetworkSize();
//        for (unsigned i = outputIdUpperBound; i < network_size; i++) {
//            prev_activation_radii[i] = show_activation(img, network, neuron_centers, i, neuron_radius);
//            circle(img, neuron_centers[i], neuron_radius, HIDDEN_COLOR, 2);
//        }
//
//    }
//
//
//    void graph_without_reallocation(int wait_miliseconds) {
//
////        for (unsigned i = 0; i < inputIdUpperBound; i++) {
////
////
////            float a = network.getNeuronById(i)->activation;
////            float color_comp = clamp(0.3 + 0.7 * a, 0.0, 1.0);
////            Scalar color = (a < 0 ? Scalar(0.3, 0.3, color_comp) : Scalar(color_comp, color_comp, color_comp)) * 255.0;
////            unsigned activation_radius = clamp(a, 0.3 , 2.0) * neuron_radius;
////            if (prev_activation_radii[i] < activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////            } else if (prev_activation_radii[i] > activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////                circle(img, neuron_centers[i], activation_radius, BLACK, prev_activation_radii[i] - activation_radius);
////            }
////
////            circle(img, neuron_centers[i], neuron_radius, INPUT_COLOR, 2);
////        }
////
////        for (unsigned i = inputIdUpperBound; i < outputIdUpperBound; i++) {
////            float a = network.getNeuronById(i)->activation;
////            float color_comp = clamp(0.3 + 0.7 * a, 0.0, 1.0);
////            Scalar color = (a < 0 ? Scalar(0.3, 0.3, color_comp) : Scalar(color_comp, color_comp, color_comp)) * 255.0;
////            unsigned activation_radius = clamp(a, 0.3 , 2.0) * neuron_radius;
////            if (prev_activation_radii[i] < activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////            } else if (prev_activation_radii[i] > activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////                circle(img, neuron_centers[i], activation_radius, BLACK, prev_activation_radii[i] - activation_radius);
////            }
////
////            circle(img, neuron_centers[i], neuron_radius, OUTPUT_COLOR, 2);
////        }
////
////        unsigned network_size = network.getNetworkSize();
////        for (unsigned i = outputIdUpperBound; i < network_size; i++) {
////            float a = network.getNeuronById(i)->activation;
////            float color_comp = clamp(0.3 + 0.7 * a, 0.0, 1.0);
////            Scalar color = (a < 0 ? Scalar(0.3, 0.3, color_comp) : Scalar(color_comp, color_comp, color_comp)) * 255.0;
////            unsigned activation_radius = clamp(a, 0.3 , 2.0) * neuron_radius;
////            if (prev_activation_radii[i] < activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////            } else if (prev_activation_radii[i] > activation_radius) {
////                circle(img, neuron_centers[i], activation_radius, color, activation_radius);
////                circle(img, neuron_centers[i], activation_radius, BLACK, prev_activation_radii[i] - activation_radius);
////            }
////
////            circle(img, neuron_centers[i], neuron_radius, HIDDEN_COLOR, 2);
////        }
//
//        for (unsigned i = 0; i < inputIdUpperBound; i++) {
//            circle(img,  neuron_centers[i], neuron_radius - 1, BLACK, -1);
//            show_activation(img, network, neuron_centers, i, neuron_radius);
//        }
//
//        for (unsigned i = inputIdUpperBound; i < outputIdUpperBound; i++) {
//            circle(img,  neuron_centers[i], neuron_radius - 1, BLACK, -1);
//            show_activation(img, network, neuron_centers, i, neuron_radius);
//        }
//
//        unsigned network_size = network->getNetworkSize();
//        for (unsigned i = outputIdUpperBound; i < network_size; i++) {
//            circle(img,  neuron_centers[i], neuron_radius - 1, BLACK, -1);
//            show_activation(img, network, neuron_centers, i, neuron_radius);
//        }
//
//        imshow("neural network", img);
//        waitKey(wait_miliseconds);
//    }
};


#endif //OPENCV_PRACTICE_GRAPHVIZ_H
