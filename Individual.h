//
// Created by WangJingjin on 2018/7/16.
//

#ifndef SRC_GENOME_H
#define SRC_GENOME_H

#include "Gene.h"
#include "Innovation.h"
#include "NeuralNetwork.h"
#include "Utils.h"
#include "Parameter.h"
#include <chrono>
#include <random>



template<class Iter, class T>
Iter binary_find(Iter begin, Iter end, T val) {
    Iter i = std::lower_bound(begin, end, val);

    if (i != end && !(val < *i)) {
        return i;
    } else {
        return end;
    }
};


class Individual {

protected:
    std::vector<ConnectionGene> connGenome;
    innov_hash_map& innov_map;
    innov_hash_map& node_innov_map;
    innov_hash_map& recurrent_innov_map;
    innov_hash_map& recurrent_node_innov_map;

    NeuralNetwork* phenotype;

    std::normal_distribution<float> &distribution;

    std::default_random_engine &generator;

    FastRandIntGenerator &intGen;
    Parameter & param;

    int &innov_front;
    int &nId_front;

public:

    std::vector<ConnectionGene>::iterator insertGene(ConnectionGene& g);

    Individual(std::vector<ConnectionGene> &genome, unsigned int inputNum, unsigned int outputNum,
               innov_hash_map &innovation_map, innov_hash_map &node_innovation_map,
               innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
               int &innovation_front, int &nodeID_front, std::normal_distribution<float> &distribution,
               std::default_random_engine &gen, FastRandIntGenerator &intGenerator, Parameter &param);


public:
    float fitness;
    const unsigned input_num;
    const unsigned output_num; //The ids of output nodes are from input_num through input_num + output_num - 1
    unsigned hidden_num;
    unsigned enabled_gene_num;
    bool alive;

    void setFitness(const float fitness);

    float getFitness() const;

    void reInit(std::vector<ConnectionGene>& newGenome);


    //TODO: start out minimally, like, 1 connection per output.
    Individual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map, innov_hash_map &node_innovation_set,
               innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
               int &innovation_front, int &nodeID_front, std::normal_distribution<float> &distribution,
               std::default_random_engine &gen, FastRandIntGenerator &intGenerator, Parameter &param);

    Individual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map, innov_hash_map &node_innovation_set,
               innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
               int &innovation_front, int &nodeID_front, std::normal_distribution<float> &distribution,
               std::default_random_engine &gen, FastRandIntGenerator &intGenerator, Parameter &param,
               bool denseInit);

    //Individual(const Individual& other) noexcept;
    Individual(Individual&& other) noexcept;

    bool operator==(const Individual& other);

    virtual ~Individual();

    const std::vector<ConnectionGene> &getConnGenome() const;
    NeuralNetwork *getPhenotype() const;

    virtual std::vector<float> evaluate(const std::vector<float>& input);
    void buildPhenotype();
    void deletePhenotype();
    void mutate_weights(const float mutation_rate, const float mutation_num_limit_ratio);
    void mutate_nodes(const float mutation_rate);

protected:
    inline void mutate_nonrecurrent_connections(const int tries);
    inline void mutate_recurrent_connections(const int tries);
private:
    void mutate_nodes_helper();
public:
    virtual void mutate_connections(const float mutation_rate, const float recur_prob, const int tries);
    virtual void mutate_reenable_genes(const float reenable_rate, const int tries);
    virtual void mutate_disable_genes(const float toggle_ratio, const int tries);

    void print_genes() const;
    std::string get_genes_info() const;

    void crossover_with(const Individual *other, Individual* placeInto);

    Individual * crossover_with(const Individual *other);

    Individual * clone();

    void flush();

    std::vector<ConnectionGene> cloneGenome();

    //Create an individual of the same type as the parent with the given genome.
    virtual Individual * create(std::vector<ConnectionGene> &genome);

    unsigned genomeSize() const;

    static void print_comparison(Individual& ind1, Individual& ind2);
    static bool checkIntegrity(const Individual* ind1);
    static void save(const char* filename, Individual* ind);
    static Individual* load(const char* filename);
};

class NonRecurrentIndividual : public Individual {

private:
NonRecurrentIndividual(std::vector<ConnectionGene> &genome, unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map,
                           innov_hash_map &node_innovation_set, innov_hash_map &recurrent_innovation_map,
                           innov_hash_map &recurrent_node_innovation_map, int &innovation_front, int &nodeID_front,
                           std::normal_distribution<float> &distribution, std::default_random_engine &gen,
                           FastRandIntGenerator &intGenerator, Parameter &param);

public:
    NonRecurrentIndividual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map,
                           innov_hash_map &node_innovation_set, innov_hash_map &recurrent_innovation_map,
                           innov_hash_map &recurrent_node_innovation_map, int &innovation_front, int &nodeID_front,
                           std::normal_distribution<float> &distribution, std::default_random_engine &gen,
                           FastRandIntGenerator &intGenerator, Parameter &param, bool denseInit);

    NonRecurrentIndividual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map,
                           innov_hash_map &node_innovation_set, innov_hash_map &recurrent_innovation_map,
                           innov_hash_map &recurrent_node_innovation_map, int &innovation_front, int &nodeID_front,
                           std::normal_distribution<float> &distribution, std::default_random_engine &gen,
                           FastRandIntGenerator &intGenerator, Parameter &param);


    virtual void mutate_connections(const float mutation_rate, const float recur_prob, const int tries = 1);
    virtual void mutate_reenable_genes(const float reenable_rate, const int tries);
    //virtual void mutate_disable_genes(const float disable_rate);

    virtual std::vector<float> evaluate(const std::vector<float>& input);

    virtual Individual * create(std::vector<ConnectionGene> &genome);

    static bool checkIntegrity(Individual* ind);
};

inline bool compare_individual (Individual*& p1, Individual*& p2) {

    if (p1->getFitness() != p2->getFitness()) {
        return p1->getFitness() > p2->getFitness();
    } else {
        return p1->getPhenotype()->getNetworkSize() < p2->getPhenotype()->getNetworkSize();
    }
}

inline bool compare_individual_for_best (Individual* const& p1, Individual* const& p2) {

    if (p1->getFitness() != p2->getFitness()) {
        return p1->getFitness() < p2->getFitness();
    } else {
        return p1->getPhenotype()->getNetworkSize() > p2->getPhenotype()->getNetworkSize();
    }
}


#endif //SRC_GENOME_H
