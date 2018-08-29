//
// Created by WangJingjin on 2018/7/16.
//

#include <cstdio>
#include <iostream>
#include <vector>
#include "Utils.h"
#include "GraphViz.h"
#include "Population.h"
#include <chrono>
#include <random>
#include <omp.h>
//#include <gperftools/heap-profiler.h>
//#include <gperftools/profiler.h>
#include "NeuralNetwork.h"


//integer& emplace(vector<integer>& v) {
//    integer a1(1);
//    a1.a = 1;
//    integer a2(2);
//    a2.a = 2;
//    v.emplace_back(a1);
//    v.emplace_back(a2);
//    return a1;
//}

inline float stddev(std::vector<unsigned> arr, float ave) {
    float sq_sum = 0.0;
    for (unsigned v : arr) {
        sq_sum += (v - ave) * (v - ave);
    }
    return sqrtf(sq_sum) / arr.size();
}

std::pair<unsigned, unsigned> run(unsigned pop_size, unsigned inputNum, unsigned outputNum, Parameter& param) {

    Population pop(pop_size, inputNum, outputNum, param, false);

    float max_fitness = 0;

    unsigned network_size = 0;

    unsigned count = 0;
    while (true) {
        ++count;
//        printf("count:%d\n", count);
//        param.mutation_rate -= count * 0.01

        std::vector<float> input(3);
        input[0] = 1.0;

//        #pragma omp parallel for schedule(static)
        for (unsigned i = 0; i < pop_size; ++i) {
            Individual* ind = pop[i];
            float fitness = 0.0;

//            ind->buildPhenotype();

            input[1] = 0;
            input[2] = 0;
            std::vector<float> outputs = ind->evaluate(input);
            fitness += (outputs[0] - 1) * (outputs[0] - 1) + outputs[1]*outputs[1] + outputs[2]*outputs[2] + outputs[3]*outputs[3];
//            fitness += outputs[0] * outputs[0];

            input[1] = 0;
            input[2] = 1;
            outputs = ind->evaluate(input);

            fitness += (outputs[0]) * (outputs[0]) + (outputs[1] - 1)*(outputs[1] - 1) + outputs[2]*outputs[2] + outputs[3]*outputs[3];
//            fitness += (outputs[0] - 1) * (outputs[0] - 1);


            input[1] = 1;
            input[2] = 0;
            outputs = ind->evaluate(input);

            fitness += outputs[0] * outputs[0] + outputs[1]*outputs[1] + (outputs[2] - 1)*(outputs[2] - 1) + outputs[3]*outputs[3];
//            fitness += (outputs[0] - 1) * (outputs[0] - 1);

            input[1] = 1;
            input[2] = 1;
            outputs = ind->evaluate(input);

            fitness += outputs[0] * outputs[0] + outputs[1]*outputs[1] + outputs[2]*outputs[2] + (outputs[3] - 1)*(outputs[3] - 1);
//            fitness += outputs[0] * outputs[0];

            ind->setFitness(1.0 / fitness);
        }
//        pop.print_info();
        if (pop.gen_count > 1000) {
            printf("generation count exceeds 1000, retry\n");
            return run(pop_size, inputNum, outputNum, param);
        }

        Individual* bestIndiv = pop.getBestIndiv();
        bestIndiv->flush();

        unsigned score = 0;

        input[1] = 0;
        input[2] = 0;
        std::vector<float> outputs = bestIndiv->evaluate(input);

        if (outputs[0] >= 0.5 && outputs[1] < 0.5 && outputs[2] < 0.5 && outputs[3] < 0.5) {
            ++score;
        }
//        if (*std::max_element(outputs.begin(), outputs.end()) == outputs[0]) {
//            ++score;
//        }
//        if (outputs[0] < 0.5) {
//            ++score;
//        }

        input[1] = 0;
        input[2] = 1;
        outputs = bestIndiv->evaluate(input);

        if (outputs[0] < 0.5 && outputs[1] >= 0.5 && outputs[2] < 0.5 && outputs[3] < 0.5) {
            ++score;
        }
//        if (*std::max_element(outputs.begin(), outputs.end()) == outputs[1]) {
//            ++score;
//        }
//        if (outputs[0] >= 0.5) {
//            ++score;
//        }

        input[1] = 1;
        input[2] = 0;
        outputs = bestIndiv->evaluate(input);

        if (outputs[0] < 0.5 && outputs[1] < 0.5 && outputs[2] >= 0.5 && outputs[3] < 0.5) {
            ++score;
        }
//        if (*std::max_element(outputs.begin(), outputs.end()) == outputs[2]) {
//            ++score;
//        }
//        if (outputs[0] >= 0.5) {
//            ++score;
//        }

        input[1] = 1;
        input[2] = 1;
        outputs = bestIndiv->evaluate(input);

        if (outputs[0] < 0.5 && outputs[1] < 0.5 && outputs[2] < 0.5 && outputs[3] >= 0.5) {
            ++score;
        }
//        if (*std::max_element(outputs.begin(), outputs.end()) == outputs[3]) {
//            ++score;
//        }
//        if (outputs[0] < 0.5) {
//            ++score;
//        }


        if (score == 4) {
            GraphViz viz(bestIndiv->getConnGenome(), bestIndiv->getPhenotype(), true);
            viz.max_width = 1200;
            viz.max_height = 600;
            viz.set();
            viz.graph(0);
//            Individual::save("best_indiv.neat", bestIndiv);
            return std::make_pair(count, bestIndiv->enabled_gene_num);
//            network_size = bestIndiv->getPhenotype()->getNetworkSize();
            break;
        }

        pop.nextGeneration();
    }

//    return network_size;
}

int main() {
//    innov_hash_map map = innov_hash_map();
//    int innov_front = 64;
//    int nID_front = 64;
//    Individual ind(30, 1, map, innov_front, nID_front);
//    const std::vector<ConnectionGene> &genome = ind.getConnGenome();
//    for (ConnectionGene g : genome) {
//        g.print_info();
//    }
//
//    ind.buildPhenotype(Binary);
//    const NeuralNetwork * net = ind.getPhenotype();
//    std::vector<float> outputs = ind.evaluate(vector<float>(30, 1));
//    for (float output : outputs) {
//        printf("%f, ", output);
//    }
////    printf("input size: %lu\n", std::vector<float>(50, 1).size());
//    net->print_info();

//    std::vector<ConnectionGene> genome = std::vector<ConnectionGene>();
//    genome.emplace_back(ConnectionGene(0, 2, 0, 0.6));
//    genome.emplace_back(ConnectionGene(2, 1, 1, 0.7));
//    //genome.emplace_back(ConnectionGene(2, 1, 2, 1));
//    genome.emplace_back(ConnectionGene(1, 2, 1, -0.5));
//    for (ConnectionGene g : genome) {
//        g.print_info();
//    }
//    const NeuralNetwork net = NeuralNetwork(genome, 1, 1, Relu, 1);
//    net.print_info();
//    for (int i = 0; i < 100; i++) {
//        printf("%dth pass\n", i);
//        std::vector<float> outputs = net.evaluate(vector<float>(1, 1));
//        for (float output : outputs) {
//            printf("%f, ", output);
//        }
//        printf("\n");
//    }

//    randGen gen;
//    printf("%f\n", gen.xorshf96() % 100 * 0.01);
//
//    unsigned ninterval = 20;
//    std::vector<unsigned> dist = std::vector<unsigned>(ninterval, 0);
//    unsigned range = 100;
//    unsigned skip = range/ninterval;
//    for (int i = 0; i < 10000; i++) {
//        int num = gen.xorshf96() % range;
//        ++dist[num / skip];
//    }
//    for (int i = 0; i < ninterval; i++) {
//        printf("%d - %d: %d\n", i*skip, (i+1)*skip, dist[i]);
//    }

//    const int nrolls = 100; // number of experiments
//    const int nstars = 100;   // maximum number of stars to distribute
//    const int max = 100;
//    const float prob = 0.2;
//
//    std::default_random_engine generator;
//    std::normal_distribution<float> dist(0.0, 1.0);
//    std::binomial_distribution<int> distribution(max, prob);
//
//    for (int i=0; i<nrolls; ++i) {
//        printf( "%f\n", dist(generator));
//        printf( "%d\n", distribution(generator));
//    }

//    unsigned random_seed = chrono::system_clock::now().time_since_epoch().count();
//    default_random_engine gen(random_seed);
//    normal_distribution<float> dist(0, 1);
//    FastRandIntGenerator intGen;
//    int inputNum = 3;
//    int outputNum = 4;
//    unsigned pop_size = 150;
//    int innov_front = inputNum * outputNum;
//    int nId_front = inputNum + outputNum;
//    innov_hash_map innov_map;
//
//    float weight_mutation_rate = 0.05;
//    float node_mutation_rate = 0.0025;
//    float connection_mutation_rate = 0.005;
//
//    const float excess_gene_factor(1.0), disjoint_gene_factor(1.0), weight_factor(0.4);
//
    unsigned pop_size = 150;
    unsigned inputNum = 3;
    unsigned outputNum = 4;

    Parameter param;
    param.speciation_threshold = 0.8;
    param.dynamic_thresholding = true;
    param.threshold_increment = 0.01;
    param.target_num_species = 5;

    param.identity_recurrent_activation = true;

    param.connection_tries = 3;
    param.weight_deviation_std = 0.25;
    param.crossover_cross_prob = 0.5;

    param.weight_mutation_rate = 0.2;
    param.mutation_num_limit_ratio = 1;
    param.node_mutation_rate = 0.025;
    param.connection_mutation_rate = 0.3;
    param.reenable_rate = 0.0;
    param.disable_rate = 0.0;
    param.recur_prob = 0.3;

    param.elitism = 0.01;
    param.elimination_num = 0.1;

    param.allow_recurrent = true;
    param.actFunc = Tanh;
    param.num_generatons_to_keep_innovation_tracking = 100;

    param.disjoint_gene_factor = 1.0;
    param.excess_gene_factor = 1.0;
    param.weight_factor = 0.4;

    param.backward_recurrency_only = true;
////
    unsigned network_size_sum = 0;
    unsigned generation_count = 0;

    float speciation_threshold = 0;

    unsigned num_iteration = 1000;

    float num_iteration_reciprocal = 1.0 / num_iteration;

    std::vector<unsigned> generation_counts;
    generation_counts.reserve(num_iteration);
    std::vector<unsigned> network_sizes;
    network_sizes.reserve(num_iteration);

    double t1 = omp_get_wtime();
//    size_t t1 = clock();

//    #pragma omp parallel for schedule(dynamic)
//    ProfilerStart("run.log");
//    HeapProfilerStart("mybin");

//    #pragma omp parallel for schedule(guided) num_threads(2)
    for (unsigned i =0; i < num_iteration; ++i) {
        std::pair<unsigned, unsigned> result = run(pop_size, inputNum, outputNum, param);

//        #pragma omp ordered
        {
            generation_count += result.first;
            network_size_sum += result.second;
            generation_counts.emplace_back(result.first);
            network_sizes.emplace_back(result.second);

            speciation_threshold += param.speciation_threshold;
        }
    }
//    HeapProfilerStop();
//    ProfilerStop();
//
    double t2 = omp_get_wtime();
//    size_t t2 = clock();

    printf("average generation count: %f\n", generation_count * num_iteration_reciprocal);
    printf("generation count std: %f\n", stddev(generation_counts, generation_count * num_iteration_reciprocal));
    printf("average network size: %f\n", network_size_sum * num_iteration_reciprocal);
    printf("network size std: %f\n", stddev(network_sizes, network_size_sum * num_iteration_reciprocal));
    printf("speciation_threshold: %f\n", speciation_threshold * num_iteration_reciprocal);

//    printf("time took: %f\n", (double)(t2 - t1) / CLOCKS_PER_SEC);
    printf("time took: %f\n", (t2 - t1));

//    Individual* ind = Individual::load("../pendulum_best_indiv_2.neat");
//    GraphViz viz(ind);
//    viz.max_width = 1200;
//    viz.max_height = 600;
//    viz.set();
//    viz.graph(0);


//
//    printf("sizeof individual: %lu\n", sizeof(Individual));
//    printf("sizeof NonRecurrentIndividual: %lu\n", sizeof(NonRecurrentIndividual));



//    Population pop(pop_size, inputNum, outputNum, param, 0, 1);
//
//    std::vector<float> input(3);
//    input[0] = 1.0;
//    for (unsigned i = 0; i < 100; i++) {
//
//        for (Individual*& ind : pop.pop) {
//            float fitness = 0.0;
//
//            input[1] = 0;
//            input[2] = 0;
//            float output = ind->evaluate(input)[0];
//            fitness += (output) * (output);
//
//            input[1] = 0;
//            input[2] = 1;
//            output = ind->evaluate(input)[0];
//
//            fitness += (output - 1) * (output - 1);
//
//            input[1] = 1;
//            input[2] = 0;
//            output = ind->evaluate(input)[0];
//
//            fitness += (output - 1) * (output - 1);
//
//            input[1] = 1;
//            input[2] = 1;
//            output = ind->evaluate(input)[0];
//
//            fitness += (output) * (output);
//
//            ind->setFitness(1.0 / fitness);
//        }
//        pop.print_info();
//
//        Individual* ind = pop.getBestIndiv();
//        ind->buildPhenotype(Relu);
//
//        GraphViz viz(ind->getConnGenome(), ind->getPhenotype(), true);
//        viz.max_width = 1200;
//        viz.max_height = 600;
//        viz.set();
//        viz.graph(1);
//
//        pop.nextGeneration();
//    }
//
//    for (Individual*& ind : pop.pop) {
//        float fitness = 0.0;
//
//        input[0] = 0;
//        input[1] = 0;
//        float output = ind->evaluate(input)[0];
//        fitness += output * output;
//
//        input[0] = 0;
//        input[1] = 1;
//        output = ind->evaluate(input)[0];
//
//        fitness += (output - 1) * (output - 1);
//
//        input[0] = 1;
//        input[1] = 0;
//        output = ind->evaluate(input)[0];
//
//        fitness += (output - 1) * (output - 1);
//
//        input[0] = 1;
//        input[1] = 1;
//        output = ind->evaluate(input)[0];
//
//        fitness += output * output;
//
//        ind->setFitness(1.0 / fitness);
 //   }
//
//    Individual* ind = pop.getBestIndiv();
//    ind->buildPhenotype(Relu);
//
//    GraphViz viz(ind->getConnGenome(), ind->getPhenotype(), true);
//    viz.max_width = 1200;
//    viz.max_height = 600;
//    viz.set();
//    viz.graph(0);

//
//    const int inputNum = 3;
//    const int outputNum = 4;
//
//
//    innov_hash_map innov_map;
//    innov_hash_map node_set;
//    innov_hash_map recurrent_innov_map;
//    innov_hash_map recurrent_node_innov_map;
//
//    innov_map.set_empty_key(pair<int, int>(-1, -1));
//    node_set.set_empty_key(pair<int, int>(-1, -1));
//    recurrent_innov_map.set_empty_key(pair<int, int>(-1, -1));
//    recurrent_node_innov_map.set_empty_key(pair<int, int>(-1, -1));
////
//    int innov_front = 12;
//    int nId_front = 7;
////
//    normal_distribution<float> dist(0.0, 1.0);
//    default_random_engine gen(chrono::system_clock::now().time_since_epoch().count());
//    FastRandIntGenerator intGen;
//////
//    const float weight_mutation_rate = 0.1;
//    const float weight_mutation_limit_ratio = 0.1;
//    const float node_mutation_rate = 0.05;
//    const float connection_mutation_rate = 0.3;
//    const float gene_reenable_rate = 0.1;
////
//    Individual* ind = new Individual(inputNum, outputNum, innov_map, node_set, recurrent_innov_map, recurrent_node_innov_map, innov_front, nId_front, dist, gen, intGen, param, false);
////
////
//////    NonRecurrentIndividual ind2 = NonRecurrentIndividual(inputNum, outputNum, innov_map, innov_front, nId_front, dist, gen, intGen, true);
//////
//////    Individual::print_comparison(ind1, ind2);
//////
//////    printf("distance: %f\n", Species::distance(ind1.getConnGenome(), ind2.getConnGenome(), excess_gene_factor, disjoint_gene_factor, weight_factor));
//////        ind.buildPhenotype(Relu);
////
////
//    ind->buildPhenotype();
//    GraphViz viz(ind->getConnGenome(), ind->getPhenotype(), true);
//    viz.max_width = 1200;
//    viz.max_height = 600;
//    viz.set();
//    viz.graph(0);
////
//    while (true) {
//        ind->buildPhenotype();
//        ind->mutate_disable_genes(0.1, 3);
//        ind->mutate_reenable_genes(0, 3);
//        ind->mutate_weights(weight_mutation_rate, weight_mutation_limit_ratio);
//        ind->mutate_connections(connection_mutation_rate, param.recur_prob, 3);
//        ind->mutate_nodes(node_mutation_rate);
//        assert(Individual::checkIntegrity(ind));
//
//        std::vector<float> o1 = ind->evaluate(vector<float>(3, 1));
//
//        NeuralNetwork * phenotype1 = new NeuralNetwork(move(*(ind->getPhenotype())));
//
//        ind->buildPhenotype();
//
//        std::vector<float> o2 = ind->evaluate(vector<float>(3, 1));
//
//        o1 = phenotype1->evaluate(vector<float>(3, 1));
//
//        o2 = ind->evaluate(vector<float>(3, 1));
//
//        printf("o1[0]: %f\n", o1[0]);
//        printf("o2[0]: %f\n", o2[0]);
//        printf("o1[1]: %f\n", o1[1]);
//        printf("o2[1]: %f\n", o2[1]);
//        printf("o1[2]: %f\n", o1[2]);
//        printf("o2[2]: %f\n", o2[2]);
//        printf("o1[3]: %f\n", o1[3]);
//        printf("o2[3]: %f\n", o2[3]);
//
//        GraphViz viz(ind->getConnGenome(), ind->getPhenotype(), true);
//        viz.max_width = 1200;
//        viz.max_height = 600;
//        viz.set();
//        viz.graph(1);
//        if (fabsf(o1[0] - o2[0]) >= 0.001 || fabsf(o1[1] - o2[1]) >= 0.001 || fabsf(o1[2] - o2[2]) >= 0.001 || fabsf(o1[3] - o2[3]) >= 0.001) {
//
//            NeuralNetwork::compare(ind->getPhenotype(), phenotype1);
//
//            GraphViz viz(ind->getConnGenome(), ind->getPhenotype(), true);
//            viz.max_width = 1200;
//            viz.max_height = 600;
//            viz.set();
//            viz.graph(0);
//
//        }
//
//        delete phenotype1;
//    }

//    Individual ind(originalInd);

//    ind.mutate_weights(weight_mutation_rate);
//    originalInd.mutate_nodes(node_mutation_rate);
//    ind.mutate_nodes(node_mutation_rate);
//    originalInd.mutate_connections(connection_mutation_rate);
//    ind.mutate_connections(connection_mutation_rate);
//
//    Individual::print_comparison(originalInd, ind);
//    binomial_distribution<int> binom(innov_front, weight_mutation_rate);
//    int numToMutate = binom(gen);
//    printf("Number of genes to mutate = %d\n", numToMutate);
}