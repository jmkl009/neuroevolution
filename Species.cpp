//
// Created by WangJingjin on 2018/7/23.
//

#include "Species.h"
//#include <google/dense_hash_set>
//#include <unordered_set>

using namespace google;

float Species::distance(const std::vector<ConnectionGene>& genome1, const std::vector<ConnectionGene>& genome2, const float excess_gene_factor,
                        const float disjoint_gene_factor, const float weight_factor) {

    float weight_dif_acc = 0;
    unsigned disjoint_gene_count = 0;
    unsigned excess_gene_count = 0;

    unsigned shared_gene_count = 0;

    auto iter1 = genome1.begin();
    auto iter2 = genome2.begin();

    while(iter1 != genome1.end() && iter2 != genome2.end()) {
        if (iter1->innov == iter2->innov) {
            weight_dif_acc += fabsf(iter1->weight - iter2->weight);
            ++iter1;
            ++iter2;
            ++shared_gene_count;
        } else if (iter1->innov < iter2->innov) {
            ++disjoint_gene_count;
            ++iter1;
        } else {
            ++disjoint_gene_count;
            ++iter2;
        }
    }

    if (shared_gene_count == 0) { // which might be the case in non dense initialization
        return 10000;
    }
  //  assert(shared_gene_count > 0);

    if (iter1 == genome1.end() && iter2 != genome2.end()) {
        do {
            ++excess_gene_count;
            ++iter2;
        } while (iter2 != genome2.end());
    } else if (iter1 != genome1.end() && iter2 == genome2.end()) {
        do {
            ++excess_gene_count;
            ++iter1;
        } while (iter1 != genome1.end());
    }

    float larger_genome_size_reciprocal = 1.0 / (genome1.size() > genome2.size() ? genome1.size() : genome2.size());

    return excess_gene_count * excess_gene_factor * larger_genome_size_reciprocal +
            disjoint_gene_count * disjoint_gene_factor * larger_genome_size_reciprocal +
            weight_dif_acc / shared_gene_count * weight_factor;
}

Species::Species(FastRandIntGenerator& intGenerator, Parameter& parameter) : species(), intGen(intGenerator), representative_individual(nullptr),  param(parameter), tournament_size(0) {

}

Species::Species(const std::vector<Individual*> * indivs, FastRandIntGenerator& intGenerator, Parameter& parameter) : species(*indivs), intGen(intGenerator), representative_individual(nullptr), param(parameter), tournament_size(0) {

}

float Species::adjustFitnesses(const float excess_gene_factor, const float disjoint_gene_factor, const float weight_factor) {
    unsigned species_size = species.size();

    float num_individuals_reciprocal = 1.0 / species.size();
    float fitness_sum = 0.0;

    for (unsigned i = 0; i < species_size; i++) {

        //TODO: Below algorithm is for an experimental way to adjust fitnesses.
//        float distance_sum = 0.0;
//        const std::vector<ConnectionGene>& genome = species[i]->getConnGenome();
//        for (unsigned j = 0; j < i; j++) {
//            distance_sum += distance(genome, species[j]->getConnGenome(), excess_gene_factor, disjoint_gene_factor, weight_factor);
//        }
//        for (unsigned j = i + 1; j < species_size; j++) {
//            distance_sum += distance(genome, species[j]->getConnGenome(), excess_gene_factor, disjoint_gene_factor, weight_factor);
//        }
//        species[i]->setFitness(species[i]->getFitness() / distance_sum);

        //This is an easy way.
        float adjusted_fitness = species[i]->getFitness() * num_individuals_reciprocal;
        species[i]->setFitness(adjusted_fitness);

        fitness_sum += adjusted_fitness;
    }
    return fitness_sum;

}

void Species::setRepresentativeIndividual() {
    representative_individual = species[intGen.xorshf96() % species.size()];
}

void Species::addIndividual(Individual *newInd) {
    species.emplace_back(newInd);
}

void Species::clear() {
    species.clear();
 //   species = std::vector<Individual*>();
}

float Species::adjustFitnesses() {
    unsigned species_size = species.size();

    float num_individuals_reciprocal = 1.0 / species.size();
    float fitness_sum = 0.0;

    for (unsigned i = 0; i < species_size; i++) {

        float adjusted_fitness = species[i]->getFitness() * num_individuals_reciprocal;
        species[i]->setFitness(adjusted_fitness);
//        printf("adjusted fitness: %f\n", adjusted_fitness);

        fitness_sum += adjusted_fitness;
    }
    return fitness_sum;
}

float Species::computeCompatibility(const Individual* other) const {
    assert(representative_individual != nullptr);

    return distance(representative_individual->getConnGenome(), other->getConnGenome(), param.excess_gene_factor, param.disjoint_gene_factor, param.weight_factor);
}

Individual *Species::generateOffspring() {

    unsigned species_size = species.size();


    if (species_size == 1) {
        return species[0]->clone();
    }

    if (species_size == 2) {
        return species[0]->fitness > species[1]->fitness ? species[0]->crossover_with(species[1]) : species[1]->crossover_with(species[0]);
//        return species[0]->fitness > species[1]->fitness ? species[0]->clone() : species[1]->clone();
    }

//    assert(tournament_size >= 2);
    //Tournament selection

    Individual* p1 = tournamentSelect();
    Individual* p2 = tournamentSelect();

//    return p1->fitness > p2->fitness ? p1->clone() : p2->clone();
    return p1->fitness > p2->fitness ? p1->crossover_with(p2) : p2->crossover_with(p1);
}

void Species::generateOffspring(Individual *placeInto) {
    unsigned species_size = species.size();


    if (species_size == 1) {
        std::vector<ConnectionGene> cloned_genome = species[0]->cloneGenome();
        placeInto->reInit(cloned_genome);
        placeInto->enabled_gene_num = species[0]->enabled_gene_num;
        placeInto->hidden_num = species[0]->getPhenotype()->getHiddenNum();
        return;
    }

    if (species_size == 2) {
        species[0]->fitness > species[1]->fitness ? species[0]->crossover_with(species[1], placeInto) : species[1]->crossover_with(species[0], placeInto);
        return;
    }

//    assert(tournament_size >= 2);
    //Tournament selection

    Individual* p1 = tournamentSelect();
    Individual* p2 = tournamentSelect();

//    return p1->fitness > p2->fitness ? p1->clone() : p2->clone();
    p1->fitness > p2->fitness ? p1->crossover_with(p2, placeInto) : p2->crossover_with(p1, placeInto);
}

Individual* Species::tournamentSelect() {
    unsigned species_size = species.size();

//    std::vector<Individual*> tournament_group;
//    tournament_group.reserve(tournament_size);

//    dense_hash_set<unsigned> selectedIdxes;
//    selectedIdxes.resize(tournament_size * 1.5);
//    selectedIdxes.max_load_factor(0.7);
//    selectedIdxes.min_load_factor(0.0);
//    selectedIdxes.set_empty_key(-1);

    unsigned min_idx = species_size - 1;

    for (unsigned i = 0; i < tournament_size; i++) {
        unsigned selectedIdx = intGen.xorshf96() % species_size;
//        while(selectedIdxes.find(selectedIdx) != selectedIdxes.end()) {
//            selectedIdx = intGen.xorshf96() % species_size;
//        }
        if (min_idx > selectedIdx) {
            min_idx = selectedIdx;
        }
//        selectedIdxes.emplace(selectedIdx);

//        tournament_group.emplace_back(species[selectedIdx]);
    }

    //Tournament begin
//    unsigned best_indiv_idx = 0;
////
////    for (unsigned i = 1; i < tournament_size; i++) {
////        float fitness1 = tournament_group[best_indiv_idx]->fitness;
////        float fitness2 = tournament_group[i]->fitness;
////        if (fitness1 < fitness2 || (fitness1 == fitness2 && tournament_group[best_indiv_idx]->enabled_gene_num > tournament_group[i]->enabled_gene_num)) {
////            best_indiv_idx = i;
////        }
////    }

//    return tournament_group[best_indiv_idx];
     return species[min_idx];
}

Species::Species(Species &other) : species(std::move(other.species)), intGen(other.intGen), representative_individual(other.representative_individual), param(other.param), tournament_size(0) {
}

Species::Species(Species &&other) : species(std::move(other.species)), intGen(other.intGen), representative_individual(other.representative_individual), param(other.param), tournament_size(0) {
}

void Species::reserve(unsigned size) {
    species.reserve(size);
}

void Species::setTournamentSize() {
    tournament_size = (unsigned)sqrtf((float)species.size());
    if (tournament_size < 2) {
        tournament_size = 2;
    }
}

void Species::clean() {
    while (!species.empty() && !species.back()->alive) {
        species.pop_back();
    }
}

unsigned Species::size() const{
    return species.size();
}

Species &Species::operator=(Species &&other) {
    species = std::move(other.species);
    intGen = other.intGen;
    representative_individual = other.representative_individual;
    param = other.param;
    tournament_size = other.tournament_size;

    other.species = std::vector<Individual*>();

    return *this;
}

void Species::sort() {
    std::sort(species.begin(), species.end(), compare_individual);
}
