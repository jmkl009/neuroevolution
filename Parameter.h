//
// Created by WangJingjin on 2018/7/16.
//

#ifndef SRC_PARAMETER_H
#define SRC_PARAMETER_H

#include "NeuralNetwork.h"

struct Parameter {
public:
    //weight_mutation_rate: the probability that an enabled connection's weight is perturbed, it is evaluated per gene.
    //node_mutation_rate: the probability that a node is added to the network, two nodes added at probability node_mutation_rate^2, and so on, it is evaluated per network.
    //connection_mutation_rate: the probability that a connection is added between previously unconnected nodes, it is evaluated per network.
    //reenable rate: the rate at which the first disabled genes gets reenabled.
    //disable ratio: the approximate ratio at which genes are disabled.
    //mutation_num_limit_ratio: the max number of weights to be perturbed in ratio to the genome size, it is used with weight_mutation_rate to bias the mutation towards the end of the genome.

    float weight_mutation_rate, node_mutation_rate, connection_mutation_rate, reenable_rate, disable_rate, mutation_num_limit_ratio;
    unsigned connection_tries;
    float recur_prob;

    float elitism, elimination_num;
    float excess_gene_factor, disjoint_gene_factor, weight_factor; //For speciation purpose
    float speciation_threshold;

    float weight_deviation_std;
    float crossover_cross_prob;

    unsigned num_generatons_to_keep_innovation_tracking;


    //Only work if dynamic_thresholding is true
    unsigned target_num_species;
    float threshold_increment;

    bool allow_recurrent;
    bool dynamic_thresholding;

    //Only evolove recurrent links that connect back through layers.
    bool backward_recurrency_only;

    bool identity_recurrent_activation;

    ActivationFunction actFunc;

    Parameter() :
            weight_mutation_rate(0.2), node_mutation_rate(0.025), connection_mutation_rate(0.3), connection_tries(3), recur_prob(0.2), reenable_rate(0.0), disable_rate(0.0), mutation_num_limit_ratio(1),
            elitism(0.01), elimination_num(0.1),
            excess_gene_factor(1.0), disjoint_gene_factor(1.0), weight_factor(0.4),
            speciation_threshold(1.0),
            weight_deviation_std(0.25),
            crossover_cross_prob(0.5),
            num_generatons_to_keep_innovation_tracking(100),
            target_num_species(5), threshold_increment(0.01),
            allow_recurrent(false),
            dynamic_thresholding(true),
            backward_recurrency_only(true),
            identity_recurrent_activation(false),
            actFunc(Tanh)
    {};
};


#endif //SRC_PARAMETER_H
