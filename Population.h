//
// Created by WangJingjin on 2018/7/16.
//

#ifndef SRC_POPULATION_H
#define SRC_POPULATION_H
#include "Individual.h"
#include "Parameter.h"
#include "Innovation.h"
#include "Utils.h"
#include "Species.h"
#include <random>
#include <list>
#include <unordered_map>
#include <vector>
#include <list>
#include <deque>
#include <boost/container/list.hpp>


class Population {
public:
    typedef std::vector<Species> SpeciesList;

//    Individual* bestIndiv;
//    float max_fitness;

    std::vector<Individual*>* pop;

    std::vector<Individual*>* pop_buf;

    SpeciesList species;

    innov_hash_map innov_map; //Cleared every once in a while
    innov_hash_map node_innov_map;
    innov_hash_map recur_innov_map;
    innov_hash_map recur_node_innov_map;

    Parameter &param;
    const unsigned input_num;
    const unsigned output_num;
    unsigned pop_size;
    unsigned tournament_size;
    int innovation_front;
    std::normal_distribution<float> distribution;

    std::default_random_engine generator;

    FastRandIntGenerator intGen;

    int nodeID_front;
    int gen_count;

    Individual* best_indiv;

    //TODO: implement speciation
private:
    void placeIntoSpecies(Individual * indiv);

public:
    Population(unsigned int popSize, const unsigned int inputNum, const unsigned int outputNum, Parameter &parameter,
               bool denseInit = true,
               unsigned random_seed = std::chrono::system_clock::now().time_since_epoch().count());


    void nextGeneration();

    void print_info() const;

    std::string get_info() const;

    void mutate(Individual* indiv);

    Individual * tournamentSelect();

    Individual *getBestIndiv() const;

    Individual * operator[](unsigned idx) const;

    ~Population();

};


#endif //SRC_POPULATION_H
