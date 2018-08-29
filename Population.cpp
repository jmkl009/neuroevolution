//
// Created by WangJingjin on 2018/7/16.
//

#include "Population.h"
#include <google/dense_hash_set>
#include <sstream>

//TODO:Don't forget to initialize the bias input
//TODO: Implement sparse initialization
Population::Population(unsigned int popSize, const unsigned int inputNum, const unsigned int outputNum,
                       Parameter &parameter,
                       bool denseInit,
                       unsigned random_seed) :
        param(parameter), input_num(inputNum), output_num(outputNum), pop(new std::vector<Individual*>), pop_buf(new std::vector<Individual*>), innov_map(), node_innov_map(), gen_count(0),
        pop_size(popSize), distribution(0, parameter.weight_deviation_std), generator(random_seed), intGen(), best_indiv(nullptr) {

    assert(popSize > 1);

    innov_map.max_load_factor(0.7);
    node_innov_map.max_load_factor(0.7);
    recur_innov_map.max_load_factor(0.7);
    recur_node_innov_map.max_load_factor(0.7);

//    innov_map.set_empty_key(pair<int, int>(-1, -1));
//    node_innov_map.set_empty_key(pair<int, int>(-1, -1));
//    recur_innov_map.set_empty_key(pair<int, int>(-1, -1));
//    recur_node_innov_map.set_empty_key(pair<int, int>(-1, -1));

    innovation_front = input_num * output_num;
    nodeID_front = input_num + output_num;
    pop->reserve(pop_size);
    pop_buf->reserve(pop_size);

    if (param.allow_recurrent) {
        for (unsigned i = 0; i < pop_size; i++) {
            pop->emplace_back(
                    new Individual(input_num, output_num, innov_map, node_innov_map, recur_innov_map, recur_node_innov_map,
                                   innovation_front, nodeID_front,
                                   distribution, generator, intGen, param, denseInit));
            pop->back()->buildPhenotype();
            pop_buf->emplace_back(
                    new Individual(input_num, output_num, innov_map, node_innov_map, recur_innov_map, recur_node_innov_map,
                                   innovation_front, nodeID_front,
                                   distribution, generator, intGen, param));
        }
    } else {
        for (unsigned i = 0; i < pop_size; i++) {
            pop->emplace_back(
                    new NonRecurrentIndividual(input_num, output_num, innov_map, node_innov_map, recur_innov_map,
                                               recur_node_innov_map,
                                               innovation_front, nodeID_front,
                                               distribution, generator,
                                               intGen, param, denseInit));
            pop->back()->buildPhenotype();
            pop_buf->emplace_back(
                    new NonRecurrentIndividual(input_num, output_num, innov_map, node_innov_map, recur_innov_map, recur_node_innov_map,
                                   innovation_front, nodeID_front,
                                   distribution, generator, intGen, param));
        }
    }


    species.emplace_back(Species(pop, intGen, param));

    tournament_size = (unsigned)sqrtf((float)pop_size);

}

void Population::nextGeneration() {
  std::sort(pop->begin(), pop->end(), compare_individual);
  //  printf("Dealing with elitism...\n");
  //Deal with elitism
  unsigned elite_num = pop_size * param.elitism;
  // printf("elite_num: %d\n", elite_num);

//    float prev_fitness = 10;
  for (unsigned i = 0; i < elite_num; i++) {
      std::vector<ConnectionGene> cloned_genome = (*pop)[i]->cloneGenome();
      (*pop_buf)[i]->reInit(cloned_genome);
      (*pop_buf)[i]->enabled_gene_num = (*pop)[i]->enabled_gene_num;
      (*pop_buf)[i]->hidden_num = (*pop)[i]->hidden_num;
      (*pop_buf)[i]->buildPhenotype();
//        assert((*pop)[i]->fitness < prev_fitness);
//        prev_fitness = (*pop)[i]->fitness;
      //    nextGen.back()->buildPhenotype(param.actFunc);
  }
//    bestIndiv = *std::max_element(pop->begin(), pop->end(), compare_individual_for_best);
//    max_fitness = bestIndiv->fitness;
    //Calculate the adjusted fitnesses
    unsigned species_size = species.size();
    std::vector<float> adjusted_fitnesses;
    adjusted_fitnesses.reserve(species_size);

    float adjusted_fitness_sum = 0.0;

    for (Species & s : species) {
        float adjusted_fitness_for_species = s.adjustFitnesses();
        adjusted_fitnesses.emplace_back(adjusted_fitness_for_species);
        adjusted_fitness_sum += adjusted_fitness_for_species;
    }

    float adjusted_fitness_sum_reciprocal = 1.0 / adjusted_fitness_sum;

//    std::vector<Individual*> nextGen;
//    nextGen.reserve(pop_size);

    std::sort(pop->begin(), pop->end(), compare_individual);

    //   printf("Eliminating the worst individuals...\n");
    //Eliminate the worst ones (In order to make some species die out)
    unsigned elimination_Idx_upper = pop_size - pop_size * param.elimination_num;
    for (unsigned i = pop_size - 1; i >= elimination_Idx_upper; i--) {
        (*pop)[i]->alive = false;
    }

    //  printf("Erasing empty species...\n");
    //Check to see if any species becomes empty, if so erase them.
    auto iter = species.begin();
    while (iter != species.end()) {
        iter->sort();
        iter->clean();
        if (iter->size() == 0) {
            iter = species.erase(iter);
        } else {
            ++iter;
        }
    }
    species_size = species.size();

    unsigned buf_idx = elite_num;

 //   printf("Reproducing on the level of species...\n");
    //Reproduce on the level of species
    unsigned num_indivs_remained_to_fill = pop_size - elite_num;

    auto species_iter = species.begin();
    for (unsigned i = 0; i < species_size; i++) {
        species_iter->setTournamentSize();

//        binomial_distribution<unsigned> binom(num_indivs_remained_to_fill, adjusted_fitnesses[i] * adjusted_fitness_sum_reciprocal);
//        unsigned num_allowed_to_reproduce = binom(generator);
        unsigned num_allowed_to_reproduce = (unsigned)(num_indivs_remained_to_fill *
                adjusted_fitnesses[i] * adjusted_fitness_sum_reciprocal);

//        assert(species_iter->size() != 0);
        for (unsigned j = 0; j < num_allowed_to_reproduce; ++j) {
//            nextGen.emplace_back(species[i].generateOffspring());
            species_iter->generateOffspring((*pop_buf)[buf_idx]);
            ++buf_idx;
        }

        ++species_iter;
    }

 //   printf("Generating the last bit of inidividuals...\n");
    //Generate the last bit of individuals using tournament selection
//    unsigned num_indivs_remained = pop_size - nextGen.size();

    //Tournament selection
    for (unsigned i = buf_idx; i < pop_size; i++) {

        //Tournament begin
        Individual* p1 = tournamentSelect();
        Individual* p2 = tournamentSelect();


        //nextGen.emplace_back(tournament_group[best_indiv_idx]->
         //       crossover_with(tournament_group[second_best_indiv_idx]));
//        nextGen.emplace_back(p1->fitness > p2->fitness ? p1->clone() : p2->clone());
//        nextGen.emplace_back(p1->fitness > p2->fitness ? p1->crossover_with(p2) : p2->crossover_with(p1));

        p1->fitness > p2->fitness ? p1->crossover_with(p2, (*pop_buf)[i]) : p2->crossover_with(p1, (*pop_buf)[i]);
    }


 //   printf("Placing the next generation into species and mutate a bit...\n");
    //Placing the next generation into species and mutate a bit;
    for (Species &s : species) { //Clearing all the species to make room for the new generation
        s.setRepresentativeIndividual();
        s.clear();
    }


//        #pragma omp parallel for ordered schedule(guided)
        for (unsigned i = 0; i < elite_num; i++) {
            placeIntoSpecies((*pop_buf)[i]);
        }
        for (unsigned i = elite_num; i < pop_size; i++) {
            Individual *curr = (*pop_buf)[i];
            mutate(curr);

//            #pragma omp ordered
            {
                placeIntoSpecies(curr);
            }

        }

    species_size = species.size();

  //  printf("Deleting unnecessary individuals from the last generation...\n");
    //Delete unnecessary individuals from the last generation
//    #pragma omp parallel for schedule(guided)
//    for (unsigned i = elite_num; i < pop_size; i++) {
//        delete (*pop)[i];
//    }

    if (param.dynamic_thresholding) {
        if (species_size < param.target_num_species) {
            param.speciation_threshold -= param.threshold_increment;
        } else if (species_size > param.target_num_species) {
            param.speciation_threshold += param.threshold_increment;
        }
    }

//    (*pop) = move(nextGen);

    //swap
    std::vector<Individual*> * temp = pop;
    pop = pop_buf;
    pop_buf = temp;

    ++gen_count;
    if (gen_count % param.num_generatons_to_keep_innovation_tracking == 0) {
        innov_map.clear();
        node_innov_map.clear();
        recur_innov_map.clear();
        recur_node_innov_map.clear();
    }

}

Individual *Population::getBestIndiv() const {

//    unsigned best_indiv_idx = 0;
//
//    for (unsigned i = 1; i < pop_size; i++) {
//        if (pop[best_indiv_idx]->fitness < pop[i]->fitness) {
//            best_indiv_idx = i;
//        }
//    }

//    return pop[best_indiv_idx];
    return *std::max_element(pop->begin(), pop->end(), compare_individual_for_best);
}



Population::~Population() {
    for (Individual *& ind : *pop) {
        delete ind;
    }

    for (Individual *& ind : *pop_buf) {
        delete ind;
    }

    delete pop;
    delete pop_buf;
}

void Population::mutate(Individual *indiv) {
    indiv->buildPhenotype();
    indiv->mutate_disable_genes(param.disable_rate, param.connection_tries);
    indiv->mutate_reenable_genes(param.reenable_rate, param.connection_tries);
    indiv->mutate_weights(param.weight_mutation_rate, param.mutation_num_limit_ratio);
    indiv->mutate_connections(param.connection_mutation_rate, param.recur_prob, param.connection_tries);
    indiv->mutate_nodes(param.node_mutation_rate);
//    assert(indiv->getPhenotype() != nullptr);
//    if (!Individual::checkIntegrity(indiv)) {
//        indiv->print_genes();
//        assert(0);
//    }
//    if (indiv->getPhenotype()->recurrentLinkExists()) {
//        indiv->print_genes();
//        assert(0);
//    }
}

Individual *Population::tournamentSelect() {

//    std::vector<Individual*> tournament_group;
//    tournament_group.reserve(tournament_size);
//
//    dense_hash_set<unsigned> selectedIdxes;
//    selectedIdxes.min_load_factor(0.0);
//    selectedIdxes.max_load_factor(0.7);
//    selectedIdxes.resize(tournament_size * 2);
//    selectedIdxes.set_empty_key(-1);

    unsigned min_idx = pop_size - 1;
    for (unsigned i = 0; i < tournament_size; i++) {
        unsigned selectedIdx = intGen.xorshf96() % pop_size;
//        while(selectedIdxes.find(selectedIdx) != selectedIdxes.end()) {
//            selectedIdx = intGen.xorshf96() % pop_size;
//        }
//        selectedIdxes.emplace(selectedIdx);

//        tournament_group.emplace_back((*pop)[selectedIdx]);

        if (min_idx > selectedIdx) {
            min_idx = selectedIdx;
        }
    }

    //Tournament begin
//    unsigned best_indiv_idx = 0;
//
//    for (unsigned i = 1; i < tournament_size; i++) {
//        float fitness1 = tournament_group[best_indiv_idx]->fitness;
//        float fitness2 = tournament_group[i]->fitness;
//        if (fitness1 < fitness2 || (fitness1 == fitness2 && tournament_group[best_indiv_idx]->enabled_gene_num > tournament_group[i]->enabled_gene_num)) {
//            best_indiv_idx = i;
//        }
//    }

//    return tournament_group[best_indiv_idx];
      return (*pop)[min_idx];
}

void Population::placeIntoSpecies(Individual *indiv) {
    bool placed_successful = false;
    for (Species &s : species) {
        if (s.computeCompatibility(indiv) < param.speciation_threshold) {
            s.addIndividual(indiv);
            return;
        }
    }

    if (!placed_successful) { //Create a new species
        species.emplace_back(Species(intGen, param));
        species.back().addIndividual(indiv);
        species.back().setRepresentativeIndividual();
    }
}

Individual *Population::operator[](unsigned idx) const {
    return (*pop)[idx];
}

void Population::print_info() const{

    printf("generation: %d\n", gen_count);

    float max_adjusted_fitness = (*pop)[0]->getFitness();
    float min_adjusted_fitness = max_adjusted_fitness;
    float fitness_sum = max_adjusted_fitness;


    for (unsigned i = 1; i < pop_size; i++) {
        float currFitness = (*pop)[i]->getFitness();
        if (max_adjusted_fitness < currFitness) {
            max_adjusted_fitness = currFitness;
        } else if (min_adjusted_fitness > currFitness) {
            min_adjusted_fitness = currFitness;
        }

        fitness_sum += currFitness;
    }

//    printf("max_fitness: %f\n", max_fitness);
    printf("max_fitness: %f\n", max_adjusted_fitness);
    printf("min_fitness: %f\n", min_adjusted_fitness);
    printf("average_adjusted_fitness: %f\n", fitness_sum / pop_size);

    printf("number of species: %lu\n", species.size());

    printf("Innovation front: %d\n", innovation_front);
    printf("Node ID front: %d\n", nodeID_front);

    unsigned species_size = species.size();

    auto iter = species.begin();
    for (unsigned i = 0; i < species_size; i++) {
        printf("%dth species size: %d\n", i + 1, iter->size());
        ++iter;
    }
}

std::string Population::get_info() const {
    std::stringstream str;
    str << "generation: " << gen_count << "\n";

    float max_adjusted_fitness = (*pop)[0]->getFitness();
    float min_adjusted_fitness = max_adjusted_fitness;
    float fitness_sum = max_adjusted_fitness;


    for (unsigned i = 1; i < pop_size; i++) {
        float currFitness = (*pop)[i]->getFitness();
        if (max_adjusted_fitness < currFitness) {
            max_adjusted_fitness = currFitness;
        } else if (min_adjusted_fitness > currFitness) {
            min_adjusted_fitness = currFitness;
        }

        fitness_sum += currFitness;
    }

//    str << "max_fitness: " << max_fitness << "\n";
    str << "max_fitness: " << max_adjusted_fitness << "\n";
    str << "min_fitness: " << min_adjusted_fitness << "\n";
    str << "average_fitness: " << fitness_sum / pop_size << "\n";
    str << "number of species: " << species.size() << "\n";
    str << "innovation front: " << innovation_front << "\n";
    str << "node id front: " << nodeID_front << "\n";

    unsigned species_size = species.size();

    auto iter = species.begin();
    for (unsigned i = 0; i < species_size; i++) {
        str << i+1 << "th species size: " << iter->size() << "\n";
        ++iter;
    }

    return str.str();
}
