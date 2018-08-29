//
// Created by WangJingjin on 2018/7/23.
//

#ifndef SRC_SPECIES_H
#define SRC_SPECIES_H

#include "Utils.h"
#include "Individual.h"
#include "Parameter.h"
#include <vector>

class Species {

private:
    std::vector<Individual*> species;

    FastRandIntGenerator& intGen;


    Individual* representative_individual; //The representative individual's index in the std::vector in population

    Parameter& param;

    unsigned tournament_size;

public:
    Species(FastRandIntGenerator& intGenerator, Parameter& parameter);

    Species(const std::vector<Individual*> * indivs, FastRandIntGenerator& intGenerator, Parameter& parameter);

    //returns the sum of adjusted fitnesses
    float adjustFitnesses(const float excess_gene_factor, const float disjoint_gene_factor, const float weight_factor);

    float adjustFitnesses();

    void setRepresentativeIndividual();

    void addIndividual(Individual* newInd);

    void reserve(unsigned size);

    void clear();

    void clean(); // erase any individual that becomes dead.

    void setTournamentSize();

    unsigned size() const;

    //The smaller, the more compatible.
    float computeCompatibility(const Individual* other) const;

    Individual* generateOffspring();

    void generateOffspring(Individual* placeInto);

    void sort();

    Individual* tournamentSelect();

    static float distance(const std::vector<ConnectionGene>& genome1, const std::vector<ConnectionGene>& genome2, const float excess_gene_factor, const float disjoint_gene_factor, const float weight_factor);

    Species(Species& other);

    Species(Species&& other);

    Species& operator=(Species&& other);

};


#endif //SRC_SPECIES_H
