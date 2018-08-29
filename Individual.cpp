//
// Created by WangJingjin on 2018/7/16.
//

#include "Individual.h"
#include <fstream>
#include <sstream>



Individual::Individual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map, innov_hash_map &node_innovation_set,
                       innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
                       int &innovation_front, int &nodeID_front, std::normal_distribution<float> &dist,
                       std::default_random_engine &gen,
                       FastRandIntGenerator &intGenerator, Parameter &param, bool denseInit) :
        connGenome(), innov_map(innovation_map), node_innov_map(node_innovation_set),
        recurrent_innov_map(recurrent_innovation_map), recurrent_node_innov_map(recurrent_node_innovation_map),
        phenotype(nullptr),
        distribution(dist), generator(gen), intGen(intGenerator),
        input_num(inputNum), output_num(outputNum), hidden_num(0),
        innov_front(innovation_front), nId_front(nodeID_front),
        param(param), alive(true), fitness(0.0) {


    int connection_num = inputNum * outputNum;
    int reserve_space = connection_num > 64 ? pow2roundup(connection_num) : 64;
    connGenome.reserve((unsigned long)reserve_space);

    int outputUpperId = inputNum + outputNum;

    if (denseInit) {
        for (int i = 0; i < inputNum; i++) {
            for (int j = inputNum; j < outputUpperId; j++) {
                connGenome.emplace_back(
                        ConnectionGene(i, j, i * outputNum + j - inputNum, distribution(generator), false));
            }
        }
        enabled_gene_num = inputNum * outputNum;
    } else { //TODO: Implement alternative initialization methods.
        for (int j = inputNum; j < outputUpperId; j++) {
            int inputID = intGen.xorshf96() % inputNum;
            connGenome.emplace_back(
                    ConnectionGene(inputID, j, inputID * outputNum + j - inputNum, distribution(generator), false));
            std::sort(connGenome.begin(), connGenome.end());
        }
        enabled_gene_num = outputNum;
    }
}


std::vector<float> Individual::evaluate(const std::vector<float>& input) {
    assert(phenotype != nullptr);
    std::vector<float> outputs = phenotype->evaluate(input);
    phenotype->tick(param.identity_recurrent_activation);
    return outputs;
}

Individual::~Individual() {
    if (phenotype != nullptr) {
        delete phenotype;
        phenotype = nullptr;
    }
}

void Individual::buildPhenotype() {
    if (phenotype != nullptr) {
        delete phenotype;
    }
    phenotype = new NeuralNetwork(connGenome, input_num, output_num, param.actFunc, hidden_num);
}

const std::vector<ConnectionGene> &Individual::getConnGenome() const {
    return connGenome;
}

NeuralNetwork *Individual::getPhenotype() const {
    return phenotype;
}

void Individual::deletePhenotype() {
    if (phenotype != nullptr) {
        delete phenotype;
        phenotype = nullptr;
    }
}

void Individual::mutate_weights(const float mutation_rate, const float mutation_num_limit_ratio) {
    unsigned genome_size = connGenome.size();

//    float new_mutation_rate = 3.0 / connGenome.size();

    std::uniform_real_distribution<int> uni_real(0, 1);
    std::binomial_distribution<unsigned> binom(genome_size, mutation_num_limit_ratio);
    unsigned num_limit = binom(generator);


    unsigned count = 0;
    for (int i = genome_size - 1; i >= 0 && count < num_limit; i--) {
        if (connGenome[i].enabled && uni_real(generator) < mutation_rate) {
            ConnectionGene& chosen_gene = connGenome[i];
            chosen_gene.weight += distribution(generator);
            if (phenotype != nullptr) {
                phenotype->updateWeight(chosen_gene.in, chosen_gene.out, chosen_gene.weight, chosen_gene.is_recurrent);
            }

            ++count;
        }
    }
}

//TODO:prefer shallower nodes to mutate
void Individual::mutate_nodes(const float mutation_rate) {
    unsigned genome_size = connGenome.size();
    std::uniform_real_distribution<float> uniform(0.0, 1.0);

    unsigned batch_size = input_num * output_num;

    if (uniform(generator) < mutation_rate) {
        ++enabled_gene_num; //2 new genes and a disabled old gene.
        if (genome_size < 5 * batch_size) { //If the genome is too small we should bias the mutation towards the older genes.


            for (unsigned i = 1; i * batch_size < genome_size; i++) { //The genes are mutated in 5 batches

                unsigned nIDToMutate = intGen.xorshf96() % batch_size + (i - 1) * batch_size;
                ConnectionGene& g = connGenome[nIDToMutate];

                if (g.enabled) {
                    int new_node_id = nId_front;
                    int new_connection_id = innov_front;

                    std::pair<int, int> new_node_pair(g.in, g.out);

                    if (!g.is_recurrent) {
                        auto node_search_result = node_innov_map.find(new_node_pair);
                        if (node_search_result == node_innov_map.end()) {
                            node_innov_map.emplace(new_node_pair, nId_front);

                            innov_map.emplace(std::make_pair(g.in, new_node_id), innov_front);
                            innov_map.emplace(std::make_pair(new_node_id, g.out), innov_front + 1);

                            innov_front += 2;
                            ++nId_front;


                        }  else {
                            new_node_id = node_search_result->second;
                            new_connection_id = innov_map.find(std::make_pair(g.in, new_node_id))->second;
                        }
                    } else {
                        auto node_search_result = recurrent_node_innov_map.find(new_node_pair);
                        if (node_search_result == recurrent_node_innov_map.end()) {
                            recurrent_node_innov_map.emplace(new_node_pair, nId_front);

                            recurrent_innov_map.emplace(std::make_pair(g.in, new_node_id), innov_front);
                            recurrent_innov_map.emplace(std::make_pair(new_node_id, g.out), innov_front + 1);

                            innov_front += 2;
                            ++nId_front;


                        }  else {
                            new_node_id = node_search_result->second;
                            new_connection_id = recurrent_innov_map.find(std::make_pair(g.in, new_node_id))->second;
                        }
                    }


                    //Adding a node
                    ConnectionGene gene1 = ConnectionGene(g.in, new_node_id, new_connection_id, 1.0, g.is_recurrent);

                    auto foundGene = binary_find(connGenome.begin(), connGenome.end(), gene1);
                    if (foundGene != connGenome.end()) { // If already exists, reenable the gene.
                        if (!foundGene->enabled) {
                            phenotype->update(*foundGene, param.actFunc);
                            // foundGene->print_info();
                            foundGene->enabled = true;

                            ++foundGene;
                            if (!foundGene->enabled) {
                                //    foundGene->print_info();
                                phenotype->update(*foundGene, param.actFunc);
                                foundGene->enabled = true;
                            }

//                        printf("reenabled gene: %d, %d, %d\n", g.in, new_node_id, g.out);
                            return;
                        }
//                    printf("gene: %d, %d, %d reenable failed because it is already enabled\n", g.in, new_node_id, g.out);
                        continue;
                    }


                    connGenome[nIDToMutate].enabled = false;

                    ConnectionGene gene2 = ConnectionGene(new_node_id, g.out, new_connection_id + 1, g.weight, g.is_recurrent);
                    insertGene(gene1);
                    insertGene(gene2);

//                printf("insert gene: %d, %d, %d\n", connGenome[nIDToMutate].in, new_node_id, connGenome[nIDToMutate].out);

                    if (phenotype != nullptr) { //TODO:Update network
                        phenotype->update(gene1, param.actFunc);
                        phenotype->update(gene2, param.actFunc);

                        phenotype->disable(connGenome[nIDToMutate]);
                    }

                    ++hidden_num;
                    return;
                }
            }

            mutate_nodes_helper();
            return;
        }

        mutate_nodes_helper();
        return;
    }

}

void Individual::mutate_nodes_helper() {
    unsigned genome_size = connGenome.size();
    while (true) {
        unsigned nIDToMutate = intGen.xorshf96() % genome_size;

        unsigned count = 0;
        while (!connGenome[nIDToMutate].enabled) {
            nIDToMutate = intGen.xorshf96() % genome_size;
            if (count >= genome_size) {
                return;
            }
        }

        ConnectionGene& selected = connGenome[nIDToMutate];

        int new_node_id = nId_front;
        int new_connection_id = innov_front;

        std::pair<int, int> new_node_pair(selected.in, selected.out);

        auto node_search_result = node_innov_map.find(new_node_pair);
        if (node_search_result == node_innov_map.end()) {
            node_innov_map.emplace(new_node_pair, nId_front);

            innov_map.emplace(std::make_pair(selected.in, new_node_id), innov_front);
            innov_map.emplace(std::make_pair(new_node_id, selected.out), innov_front + 1);

            innov_front += 2;
            ++nId_front;
        }  else {
            new_node_id = node_search_result->second;
            new_connection_id = innov_map.find(std::make_pair(selected.in, new_node_id))->second;
        }


        //Adding a node
        ConnectionGene gene1 = ConnectionGene(selected.in, new_node_id, new_connection_id, 1.0, selected.is_recurrent);

        //Reenable the gene if it is disabled
        auto foundGene = binary_find(connGenome.begin(), connGenome.end(), gene1);
        if (foundGene != connGenome.end()) { // If already exists, reenable the gene.
            if (!foundGene->enabled) {
                phenotype->update(*foundGene, param.actFunc);
                foundGene->enabled = true;

                ++foundGene;
                if (!foundGene->enabled) {
                    phenotype->update(*foundGene, param.actFunc);
                    foundGene->enabled = true;
                }
//                printf("reenabled gene: %d, %d, %d\n", gene1.in, new_node_id, foundGene->out);
                return;
            }
            continue;
        }

        selected.enabled = false;


        ConnectionGene gene2 = ConnectionGene(new_node_id, selected.out, new_connection_id + 1, selected.weight, selected.is_recurrent);

        insertGene(gene1);
        insertGene(gene2);

        if (phenotype != nullptr) { //TODO:Update network
            phenotype->update(gene1, param.actFunc);
            phenotype->update(gene2, param.actFunc);

            phenotype->disable(connGenome[nIDToMutate]);
        }

//        printf("insert gene: %d, %d, %d\n", gene1.in, new_node_id, gene2.out);

        ++hidden_num;
        return;
    }
}

//TODO: take care of the non-recurrent case, if possible.
//call mutate_connection() just after building the phenotype before evaluation to minimize memory use.
void Individual::mutate_connections(const float mutation_rate, const float recur_prob, const int tries) {
    assert(phenotype != nullptr);
    std::uniform_real_distribution<float> uniform(0.0, 1.0);
 //   std::vector<int> nIDs = mapExt::Keys_sorted(phenotype->getIdToNeuron());
    if (uniform(generator) < mutation_rate) {
        if (uniform(generator) < recur_prob) {
            mutate_recurrent_connections(tries);
        } else {
            mutate_nonrecurrent_connections(tries);
        }
    }
}

void Individual::mutate_reenable_genes(const float reenable_rate, const int tries) {
    assert(phenotype != nullptr);

    unsigned genome_size = connGenome.size();
    std::uniform_real_distribution<float> uni_real(0, 1);

    if (uni_real(generator) < reenable_rate) {
        unsigned failCount = 0;
        while (failCount < tries) {
            unsigned chosen_id = intGen.xorshf96() % genome_size;
            if (connGenome[chosen_id].enabled) {
                ++failCount;
                continue;
            }
            ConnectionGene &g = connGenome[chosen_id];
//            printf("reenabling a gene. in: %d, out:%d\n", g.in, g.out);

            if (!g.is_recurrent && phenotype->testBackwardRecurrency(g.in, g.out)) {
                ++failCount;
                continue;
            }

            g.enabled = true;
            phenotype->update(g, param.actFunc);
            ++enabled_gene_num;
            return;
        }
    }
}

void Individual::print_genes() const {
    unsigned size = connGenome.size();
    for (int i = 0; i < size; i++) {
        printf("Gene number: %d\n", i);
        connGenome[i].print_info();
        printf("\n");
    }
}

std::string Individual::get_genes_info() const {
    std::stringstream str;
    unsigned size = connGenome.size();
    for (int i = 0; i < size; i++) {
        str << "Gene number: " << i << "\n";
        str << connGenome[i].get_info() << "\n";
    }
    return str.str();
}

//Individual::Individual(const Individual &other) noexcept :
//        input_num(other.input_num), output_num(other.output_num), hidden_num(other.hidden_num), innov_map(other.innov_map),
//        innov_front(other.innov_front), nId_front(other.nId_front), connGenome(other.connGenome), phenotype(nullptr),
//        distribution(other.distribution), generator(other.generator), intGen(other.intGen), actFunc(other.actFunc) {
//
//        if (other.phenotype != nullptr) {
//            buildPhenotype(actFunc);
//        }
//}

void Individual::print_comparison(Individual &ind1, Individual &ind2) {
    printf("Individual 1 genome size:%lu\n", ind1.connGenome.size());
    printf("Individual 2 genome size:%lu\n\n", ind2.connGenome.size());

    std::vector<ConnectionGene> differing_genes;
    std::vector<ConnectionGene> disjoint_genes_from_ind1;
    std::vector<ConnectionGene> disjoint_genes_from_ind2;

 //   unsigned minUpperBound = min(ind1.connGenome.size(), ind2.connGenome.size());

    auto ind1_genome_iter = ind1.connGenome.begin();
    auto ind2_genome_iter = ind2.connGenome.begin();
    while(ind1_genome_iter != ind1.connGenome.end() && ind2_genome_iter != ind2.connGenome.end()) {
        if (ind1_genome_iter->innov == ind2_genome_iter->innov) {
            if (ind1_genome_iter->weight != ind2_genome_iter->weight) {
                differing_genes.emplace_back(*ind1_genome_iter);
                differing_genes.emplace_back(*ind2_genome_iter);
            }
            ++ind1_genome_iter;
            ++ind2_genome_iter;
        } else if (ind1_genome_iter->innov < ind2_genome_iter->innov) {
            disjoint_genes_from_ind1.emplace_back(*ind1_genome_iter);
            ++ind1_genome_iter;
        } else {
            disjoint_genes_from_ind2.emplace_back(*ind2_genome_iter);
            ++ind2_genome_iter;
        }
    }

    printf("-------------differing genes report-------------\n");

    unsigned differing_gene_num = differing_genes.size();
    if (differing_gene_num > 0) {
        printf("The number of differing gene is %d\n", differing_gene_num / 2);
        printf("Printed are the first several differing genes (up to 10)\n\n");
        for (unsigned i = 0; i < std::min(differing_gene_num, (unsigned)20); i+=2) {
            printf("----The %dth gene---- \n", i / 2 + 1);
            printf("----individual 1---- \n");
            differing_genes[i].print_info();
            printf("\n");
            printf("----individual 2---- \n");
            differing_genes[i + 1].print_info();
            printf("\n");
        }
        printf("\n");
    } else {
        printf("No differing genes were found\n\n");
    }

    printf("-------------disjoint genes from individual 1 report-------------\n");
    unsigned disjoint_gene_num_1 = disjoint_genes_from_ind1.size();
    if (disjoint_gene_num_1 > 0) {
        printf("The number of disjoint gene is %d\n", disjoint_gene_num_1);
        printf("Printed are the first several disjoint genes (up to 10)\n\n");
        for (unsigned i = 0; i < std::min(disjoint_gene_num_1, (unsigned)10); i++) {
            printf("----The %dth gene---- \n", i+1);
            disjoint_genes_from_ind1[i].print_info();
            printf("\n");
        }
        printf("\n");
    } else {
        printf("No disjoint genes from individual 1 were found.\n\n");
    }

    printf("-------------disjoint genes from individual 2 report-------------\n");
    unsigned disjoint_gene_num_2 = disjoint_genes_from_ind2.size();
    if (disjoint_gene_num_2 > 0) {
        printf("The number of disjoint gene is %d\n", disjoint_gene_num_2);
        printf("Printed are the first several disjoint genes (up to 10)\n\n");
        for (unsigned i = 0; i < std::min(disjoint_gene_num_2, (unsigned)10); i++) {
            printf("----The %dth gene---- \n", i+1);
            disjoint_genes_from_ind2[i].print_info();
            printf("\n");
        }
        printf("\n");
    } else {
        printf("No disjoint genes from individual 2 were found.\n\n");
    }

    printf("-------------excess genes report-------------\n");

    std::vector<ConnectionGene> excess_genes;

    if(ind1_genome_iter == ind1.connGenome.end() && ind2_genome_iter != ind2.connGenome.end()) {
        printf("Excess genes belong to individual 2\n\n");
        while (ind2_genome_iter != ind2.connGenome.end()) {
            excess_genes.emplace_back(*ind2_genome_iter);
            ++ind2_genome_iter;
        }
    } else if (ind1_genome_iter != ind1.connGenome.end() && ind2_genome_iter == ind2.connGenome.end()) {
        printf("Excess genes belong to individual 1\n\n");
        while (ind1_genome_iter != ind1.connGenome.end()) {
            excess_genes.emplace_back(*ind1_genome_iter);
            ++ind1_genome_iter;
        }
    } else {
        printf("No excess genes found\n\n");
    }

    unsigned excess_size = excess_genes.size();
    if (excess_size > 0) {
        printf("The number of excess gene is %d\n", excess_size);
        printf("Printed are the first several excess genes (up to 10)\n\n");
        for (unsigned i = 0; i < std::min(excess_size, (unsigned)10); i++) {
            printf("----The %dth gene---- \n", i+1);
            excess_genes[i].print_info();
            printf("\n");
        }
        printf("\n");
    }
}

Individual::Individual(Individual &&other) noexcept
        : connGenome(std::move(other.connGenome)),
          innov_map(other.innov_map), node_innov_map(other.node_innov_map),
          recurrent_innov_map(other.recurrent_innov_map), recurrent_node_innov_map(other.recurrent_node_innov_map),
          phenotype(other.phenotype), distribution(other.distribution),
          generator(other.generator), intGen(other.intGen),
          input_num(other.input_num), output_num(other.output_num), hidden_num(other.hidden_num),
          innov_front(other.innov_front), nId_front(other.nId_front), param(other.param), alive(other.alive),
          enabled_gene_num(other.enabled_gene_num) {
    other.connGenome = std::vector<ConnectionGene>();
    other.phenotype = nullptr;
}


Individual::Individual(std::vector<ConnectionGene> &genome, unsigned inputNum, unsigned outputNum,
                       innov_hash_map &innovation_map, innov_hash_map &node_innovation_map,
                       innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
                       int &innovation_front, int &nodeID_front, std::normal_distribution<float> &distribution,
                       std::default_random_engine &gen, FastRandIntGenerator &intGenerator, Parameter &param)
        : input_num(inputNum), output_num(outputNum), hidden_num(0), innov_map(innovation_map), node_innov_map(node_innovation_map),
          recurrent_innov_map(recurrent_innovation_map), recurrent_node_innov_map(recurrent_node_innovation_map),
          innov_front(innovation_front), nId_front(nodeID_front), connGenome(std::move(genome)), phenotype(nullptr),
          distribution(distribution), generator(gen), intGen(intGenerator), fitness(0.0), param(param), alive(true) {

    genome = std::vector<ConnectionGene>();

    enabled_gene_num = 0;
    for (ConnectionGene &g : connGenome) {
        if (g.enabled) {
            ++enabled_gene_num;
        }
    }
}


void Individual::crossover_with(const Individual *other, Individual *placeInto) {
    std::vector<ConnectionGene> new_genome;
    const std::vector<ConnectionGene>& other_connGenome = other->connGenome;

    new_genome.reserve(connGenome.size());

    auto connGenome_iter = connGenome.begin();
    auto other_connGenome_iter = other_connGenome.begin();

    auto connGenome_end = connGenome.end();
    auto other_connGenome_end = other_connGenome.end();

    unsigned enabled_gene_num = 0;

    std::uniform_real_distribution<float> dist(0.0, 1.0);

//
//    float fitness_sum = fitness + other->fitness;
//
//    float f1_ratio = fitness / fitness_sum;
//    float f2_ratio = 1 - f1_ratio;

    while (connGenome_iter != connGenome_end && other_connGenome_iter != other_connGenome_end) {
        if (connGenome_iter->innov == other_connGenome_iter->innov) {

            if(connGenome_iter->enabled == other_connGenome_iter->enabled) {
//
                if (dist(generator) < param.crossover_cross_prob) {
                    new_genome.emplace_back(intGen.rand_bool() ? *connGenome_iter : *other_connGenome_iter);
//                    new_genome.emplace_back(*connGenome_iter);
                } else {
                    ConnectionGene g = *connGenome_iter;
                    g.weight = (connGenome_iter->weight+ other_connGenome_iter->weight) * 0.5;
                    new_genome.emplace_back(g);
                }

                if (connGenome_iter->enabled) {
                    ++enabled_gene_num;
                }

            } else { //If we don't care about reenabling genes, this part can be eliminated.
//                ConnectionGene new_gene = intGen.rand_bool() ? *the_more_fit_genome_iter : *the_less_fit_genome_iter;
//                if(unireal_random(gen) < gene_reenabled_rate) {
//                    new_gene.enabled = true;
//                }
                new_genome.emplace_back(*connGenome_iter);

                if (connGenome_iter->enabled) {
                    ++enabled_gene_num;
                }
            }

            ++connGenome_iter;
            ++other_connGenome_iter;
        } else if (connGenome_iter->innov < other_connGenome_iter->innov) {

            new_genome.emplace_back(*connGenome_iter);

            if (connGenome_iter->enabled) {
                ++enabled_gene_num;
            }

            ++connGenome_iter;
        } else {
            ++other_connGenome_iter;
        }
    }

    if (connGenome_iter != connGenome_end) {
        do {
            new_genome.emplace_back(*connGenome_iter);

            if (connGenome_iter->enabled) {
                ++enabled_gene_num;
            }

            ++connGenome_iter;
        } while(connGenome_iter != connGenome_end);
    }

    placeInto->reInit(new_genome);
    placeInto->enabled_gene_num = enabled_gene_num;
    placeInto->hidden_num = phenotype->getHiddenNum();
}

Individual * Individual::crossover_with(const Individual *other) {

//    unsigned random_seed = chrono::system_clock::now().time_since_epoch().count();
    //uniform_real_distribution<float> unireal_random(0.0, 1.0);
   // default_random_engine gen(random_seed);

    std::vector<ConnectionGene> new_genome;

//    const std::vector<ConnectionGene>& the_more_fit_genome = fitness < other->fitness ? other->connGenome :
//            (fitness == other->fitness ? (connGenome.size() < other->connGenome.size() ? connGenome :
//                                          other->connGenome) : connGenome);
//    const std::vector<ConnectionGene>& the_less_fit_genome = fitness < other->fitness ? connGenome :
//            (fitness == other->fitness ? (connGenome.size() < other->connGenome.size() ? other->connGenome :
//                                          connGenome) : other->connGenome);

    const std::vector<ConnectionGene>& other_connGenome = other->connGenome;

    new_genome.reserve(connGenome.size());

    auto connGenome_iter = connGenome.begin();
    auto other_connGenome_iter = other_connGenome.begin();

    auto connGenome_end = connGenome.end();
    auto other_connGenome_end = other_connGenome.end();


    while (connGenome_iter != connGenome_end && other_connGenome_iter != other_connGenome_end) {
        if (connGenome_iter->innov == other_connGenome_iter->innov) {

            if(connGenome_iter->enabled == other_connGenome_iter->enabled) {

                new_genome.emplace_back(intGen.rand_bool() ? *connGenome_iter : *other_connGenome_iter);

            } else { //If we don't care about reenabling genes, this part can be eliminated.
//                ConnectionGene new_gene = intGen.rand_bool() ? *the_more_fit_genome_iter : *the_less_fit_genome_iter;
//                if(unireal_random(gen) < gene_reenabled_rate) {
//                    new_gene.enabled = true;
//                }
                new_genome.emplace_back(*connGenome_iter);
            }

            ++connGenome_iter;
            ++other_connGenome_iter;
        } else if (connGenome_iter->innov < other_connGenome_iter->innov) {

            new_genome.emplace_back(*connGenome_iter);

            ++connGenome_iter;
        } else {
            ++other_connGenome_iter;
        }
    }

    if (connGenome_iter != connGenome_end) {
        do {
            new_genome.emplace_back(*connGenome_iter);
            ++connGenome_iter;
        } while(connGenome_iter != connGenome_end);
    }

    return this->create(new_genome);
}

float Individual::getFitness() const {
    return fitness;
}

void Individual::flush() {
    phenotype->flush();
}

Individual * Individual::create(std::vector<ConnectionGene> &genome) {

    return new Individual(genome, input_num, output_num, innov_map, node_innov_map,
            recurrent_innov_map, recurrent_node_innov_map, innov_front, nId_front, distribution, generator, intGen, param);
}

void Individual::setFitness(const float fitness) {
    this->fitness = fitness;
}

Individual *Individual::clone() {
    std::vector<ConnectionGene> cloned_genome = connGenome;
    return this->create(cloned_genome);
}

unsigned Individual::genomeSize() const {
    return connGenome.size();
}

void Individual::mutate_disable_genes(const float toggle_ratio, const int tries) {

    unsigned genome_size = connGenome.size();
    std::uniform_real_distribution<float> uni_real(0, 1);

    if (uni_real(generator) < toggle_ratio) {
        unsigned count = 0;
        while (count < tries) {
            unsigned chosen_id = intGen.xorshf96() % genome_size;
            if (!connGenome[chosen_id].enabled) {
               // chosen_id = intGen.xorshf96() % genome_size;
                ++count;
                continue;
            }
            ConnectionGene &g = connGenome[chosen_id];
//            printf("disabling a gene. in: %d, out:%d\n", g.in, g.out);

            NeuralNetwork::OutputNeuron* out_node = (NeuralNetwork::OutputNeuron*)phenotype->getNeuronById(g.out);
            NeuralNetwork::Neuron* in_node = phenotype->getNeuronById(g.in);
            if (!g.is_recurrent && (out_node->inputs.size() == 1 || in_node->outgoings.size() == 1)) {
//                printf("disabling failed due to empty input.");
                ++count;
                continue;
            } else {
                g.enabled = false;
                phenotype->disable(g);
            }
            return;
        }

    }

//    int numToMutate = binom(generator);
//
//    for (unsigned i = 0; i < numToMutate; i++) {
//        ConnectionGene &g = connGenome[intGen.xorshf96() % genome_size];
//        if (g.enabled) {
//            g.enabled = false;
//            printf("disabled a gene.\n");
//            if (phenotype != nullptr) {
//                phenotype->disable(g.in, g.out);
//            }
//        }
//        else {
//            g.enabled = true;
//            if (phenotype != nullptr) {
//                phenotype->updateWeight(g, actFunc);
//            }
//        }
 //   }
}

bool Individual::checkIntegrity(const Individual *ind1) {
    const std::vector<ConnectionGene> &genome = ind1->getConnGenome();
    const unsigned genome_size = genome.size();

    int prev_innov = genome[0].innov;
    int curr_innov = 0;
    for (unsigned i = 1; i < genome_size; i++) {
        curr_innov = genome[i].innov;
        if (prev_innov >= curr_innov) {
            return false;
        }
        prev_innov = curr_innov;
    }
    return true;
}

std::vector<ConnectionGene>::iterator Individual::insertGene(ConnectionGene &g) {
    auto iter = std::lower_bound(connGenome.begin(), connGenome.end(), g);
    connGenome.insert(iter, g);
    return iter;
}

NonRecurrentIndividual::NonRecurrentIndividual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map,
                                               innov_hash_map &node_innovation_set,
                                               innov_hash_map &recurrent_innovation_map,
                                               innov_hash_map &recurrent_node_innovation_map, int &innovation_front,
                                               int &nodeID_front,
                                               std::normal_distribution<float> &distribution, std::default_random_engine &gen,
                                               FastRandIntGenerator &intGenerator, Parameter &param,
                                               bool denseInit)
        : Individual(
        inputNum, outputNum, innovation_map, node_innovation_set, recurrent_innovation_map, recurrent_node_innovation_map, innovation_front,
        nodeID_front,
        distribution, gen, intGenerator, param, denseInit) {

}


void Individual::mutate_nonrecurrent_connections(const int tries) {
    //   printf("linkable neuron size: %lu\n", linkableNeurons.size());
    //   std::vector<int> nIDs = mapExt::Keys_sorted(phenotype->getIdToNeuron());

    //  float rand = uniform(generator);
    //  printf("generated rand: %f \n", rand);


    std::vector<NeuralNetwork::Neuron*> linkableNeurons = phenotype->getLinkableNeurons();
    unsigned network_size = phenotype->getNetworkSize();
    unsigned output_id_upper = input_num + output_num;

    unsigned linkable_neurons_size = linkableNeurons.size();
//    printf("linkable neuron size: %u\n", linkable_neurons_size);

    int failCount = 0;
    if (linkable_neurons_size > 0) {
        while (failCount < tries) {

            NeuralNetwork::Neuron* neuronToMutate = linkableNeurons[intGen.xorshf96() % linkable_neurons_size];

            unsigned neuron_to_connect_idx = intGen.xorshf96() % network_size;

//            if (param.output_independence) { //since non-recurrent individual should not have recurrent links, we don't connect anything from the output, including itself.
                while (neuron_to_connect_idx >= input_num && neuron_to_connect_idx < output_id_upper) {
                    neuron_to_connect_idx = intGen.xorshf96() % network_size;
                }
//            }


            NeuralNetwork::Neuron* neuronToConnect = phenotype->getNeuronByIdx(neuron_to_connect_idx);

            //    assert(neuronToConnect->id >= output_id_upper || neuronToConnect->id < input_num);

            if (phenotype->testBackwardRecurrency(neuronToConnect->id, neuronToMutate->id)) {
                ++failCount;
//                printf("Abandoned this mutation due to recurrent links.\n");
                continue;
            }

//            printf("From %d connecting to %d\n", neuronToConnect->id, neuronToMutate->id);

            if (((NeuralNetwork::OutputNeuron *) neuronToMutate)->isConnectedTo(neuronToConnect, false)) {
//                printf("Abandoned this mutation due to already existent connection.\n", neuronToConnect->id, neuronToMutate->id);
                ++failCount;
                continue;
            } else {

                const int neuron_to_connect_id = neuronToConnect->id;
                const int neuron_to_mutate_id = neuronToMutate->id;

                std::pair<int, int> newPair(neuron_to_connect_id, neuron_to_mutate_id);
                int innov = innov_front;
                auto searchRes = innov_map.find(newPair);
                if (searchRes == innov_map.end()) { //Does not exist before in the same generation
                    innov_map.emplace(newPair, innov_front);
                    ++innov_front;
                } else if (neuron_to_connect_id < input_num && neuron_to_mutate_id < output_id_upper) {
                    innov = neuron_to_connect_id * output_num + neuron_to_mutate_id - input_num;
                } else {
                    innov = searchRes->second;
                }

                ConnectionGene newGene(neuron_to_connect_id, neuron_to_mutate_id, innov, distribution(generator), false);
//                ConnectionGene newGene(neuron_to_connect_id, neuron_to_mutate_id, innov, 0.0, false);

                auto foundGene = binary_find(connGenome.begin(), connGenome.end(), newGene);
                if (foundGene != connGenome.end()) { // If already exists, reenable the gene.
                    if (!foundGene->enabled) {
                        phenotype->update(*foundGene, param.actFunc);
//                        foundGene->weight = 0.0;
                        foundGene->enabled = true;
//                        printf("reenabled a past gene the same to this one\n");
                        ++enabled_gene_num;
                    }
                    ++failCount;
                    continue;
                }


                phenotype->update(newGene, param.actFunc);

                insertGene(newGene);
                ++enabled_gene_num;

                return;
            }

        }
    }
}

void Individual::mutate_recurrent_connections(const int tries) {
    std::vector<NeuralNetwork::Neuron*> linkableNeurons = phenotype->getRecurrentLinkableNeurons();
    unsigned network_size = phenotype->getNetworkSize();
    unsigned output_id_upper = input_num + output_num;

    unsigned linkable_neurons_size = linkableNeurons.size();

    //printf("linkable neuron size: %lu\n", linkable_neurons_size);

    if (linkable_neurons_size > 0) {
        int failCount = 0;
        while (failCount < tries) {

            NeuralNetwork::Neuron* neuronToMutate = linkableNeurons[intGen.xorshf96() % linkable_neurons_size];

            const int neuron_to_mutate_id = neuronToMutate->id;

            unsigned neuron_to_connect_idx = intGen.xorshf96() % network_size;
            //TODO: Think about how output independence apply in recurrent networks
//            if (param.output_independence) {
                while (neuron_to_connect_idx >= input_num && neuron_to_connect_idx < output_id_upper && neuron_to_connect_idx != neuron_to_mutate_id) {
                    neuron_to_connect_idx = intGen.xorshf96() % network_size;
                }
//            }

            NeuralNetwork::Neuron* neuronToConnect = phenotype->getNeuronByIdx(neuron_to_connect_idx);

            const int neuron_to_connect_id = neuronToConnect->id;

            if (param.backward_recurrency_only && !phenotype->testBackwardRecurrency(neuron_to_connect_id,
                                                                                     neuron_to_mutate_id)) {
                ++failCount;
                continue;
            }

            if (((NeuralNetwork::OutputNeuron *) neuronToMutate)->isConnectedTo(neuronToConnect, true)) {
//                printf("ID to connect: %d, Connection already exists.\n", neuronToConnect->id);
                failCount++;
            } else {
//                printf("From %d connecting to %d\n", neuron_to_connect_id, neuron_to_mutate_id);

                std::pair<int, int> newPair(neuron_to_connect_id, neuron_to_mutate_id);
                int innov = innov_front;
                auto searchRes = recurrent_innov_map.find(newPair);
                if (searchRes == recurrent_innov_map.end()) { //Does not exist before in the same generation
                    recurrent_innov_map.emplace(newPair, innov_front);
                    ++innov_front;
                } else {
                    innov = searchRes->second;
                }

                //TODO: fix this
                ConnectionGene newGene(neuron_to_connect_id, neuron_to_mutate_id, innov, distribution(generator), true);

                //Reenable the gene if it is disabled
                auto foundGene = binary_find(connGenome.begin(), connGenome.end(), newGene);
                if (foundGene != connGenome.end()) { // If already exists, reenable the gene.
                    if (!foundGene->enabled){
                        phenotype->update(*foundGene, param.actFunc);
                        foundGene->enabled = true;
                        ++enabled_gene_num;
                        return;
                    } else {
                        ++failCount;
                        continue;
                    }
                }


                phenotype->update(newGene, param.actFunc);

                insertGene(newGene);
                ++enabled_gene_num;

                return;
            }

        }
    }
}

bool Individual::operator==(const Individual &other) {
    unsigned genome_size = connGenome.size();
    for (unsigned i = 0; i < genome_size; i++) {
        if (connGenome[i].weight != other.connGenome[i].weight) {
            return false;
        }
    }
    return true;
}

void Individual::reInit(std::vector<ConnectionGene> &newGenome) {
    connGenome.swap(newGenome);
    alive = true;
    fitness = 0.0;
}

Individual::Individual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map, innov_hash_map &node_innovation_set,
                       innov_hash_map &recurrent_innovation_map, innov_hash_map &recurrent_node_innovation_map,
                       int &innovation_front, int &nodeID_front, std::normal_distribution<float> &distribution,
                       std::default_random_engine &gen, FastRandIntGenerator &intGenerator, Parameter &param) :
        connGenome(), innov_map(innovation_map), node_innov_map(node_innovation_set),
        recurrent_innov_map(recurrent_innovation_map), recurrent_node_innov_map(recurrent_node_innovation_map),
        phenotype(nullptr),
        distribution(distribution), generator(gen), intGen(intGenerator),
        input_num(inputNum), output_num(outputNum), hidden_num(0),
        innov_front(innovation_front), nId_front(nodeID_front),
        param(param), alive(true), fitness(0.0)  {

}

std::vector<ConnectionGene> Individual::cloneGenome() {
    return std::vector<ConnectionGene>(connGenome);
}

void NonRecurrentIndividual::mutate_connections(const float mutation_rate, const float recur_prob, const int tries) {
    std::uniform_real_distribution<float> uniform(0.0, 1.0);

    if (uniform(generator) < mutation_rate) {
        Individual::mutate_nonrecurrent_connections(tries);
    }

}

NonRecurrentIndividual::NonRecurrentIndividual(std::vector<ConnectionGene> &genome, unsigned inputNum, unsigned outputNum,
                                               innov_hash_map &innovation_map,
                                               innov_hash_map &node_innovation_set,
                                               innov_hash_map &recurrent_innovation_map,
                                               innov_hash_map &recurrent_node_innovation_map, int &innovation_front,
                                               int &nodeID_front,
                                               std::normal_distribution<float> &distribution, std::default_random_engine &gen,
                                               FastRandIntGenerator &intGenerator, Parameter &param)
        : Individual(genome, inputNum, outputNum, innovation_map, node_innovation_set, recurrent_innovation_map, recurrent_node_innovation_map,
innovation_front, nodeID_front, distribution, gen, intGenerator, param) {

}

Individual * NonRecurrentIndividual::create(std::vector<ConnectionGene> &genome) {
    return new NonRecurrentIndividual(genome, input_num, output_num, innov_map, node_innov_map, recurrent_innov_map,
                                      recurrent_node_innov_map,
                                      innov_front, nId_front, distribution, generator, intGen, param);
}

void NonRecurrentIndividual::mutate_reenable_genes(const float reenable_rate, const int tries) {
//    assert(phenotype != nullptr);
//
//    uniform_real_distribution<float> uni_real(0.0, 1.0);
//
//    if (uni_real(generator) < reenable_rate) {
//
//        for (ConnectionGene &g : connGenome) {
//            if (!g.enabled) {
//                g.enabled = true;
//                phenotype->updateWeight(g, actFunc);
//                if (phenotype->testBackwardRecurrency(g.out)) {
//                    phenotype->disable(g.in, g.out);
//                    continue;
//                } else {
//                    return;
//                }
//            }
//        }
//    }

    assert(phenotype != nullptr);

    unsigned genome_size = connGenome.size();
    std::uniform_real_distribution<float> uni_real(0, 1);

    if (uni_real(generator) < reenable_rate) {
        unsigned failCount = 0;
        while (failCount < tries) {
            unsigned chosen_id = intGen.xorshf96() % genome_size;
            if (connGenome[chosen_id].enabled) {
                ++failCount;
                continue;
            }
            ConnectionGene &g = connGenome[chosen_id];
//            printf("reenabling a gene. in: %d, out:%d\n", g.in, g.out);

            if (phenotype->testBackwardRecurrency(g.in, g.out)) {
//                printf("reenabled the gene failed due to recurrency.\n");
                ++failCount;
                continue;
            } else {
                g.enabled = true;
                phenotype->update(g, param.actFunc);
                ++enabled_gene_num;
                return;
            }
        }
    }
}

bool NonRecurrentIndividual::checkIntegrity(Individual *ind) {
    const std::vector<ConnectionGene> &genome = ind->getConnGenome();
    const unsigned genome_size = genome.size();

    int prev_innov = genome[0].innov;
    int curr_innov = 0;
    for (unsigned i = 1; i < genome_size; i++) {
        if (genome[i].is_recurrent) {
            return false;
        }
        curr_innov = genome[i].innov;
        if (prev_innov >= curr_innov) {
            return false;
        }
        prev_innov = curr_innov;
    }
    return true;
}

std::vector<float> NonRecurrentIndividual::evaluate(const std::vector<float> &input) {
    assert(phenotype != nullptr);
    std::vector<float> outputs = phenotype->evaluate(input);
    return outputs;
}

NonRecurrentIndividual::NonRecurrentIndividual(unsigned inputNum, unsigned outputNum, innov_hash_map &innovation_map,
                                               innov_hash_map &node_innovation_set,
                                               innov_hash_map &recurrent_innovation_map,
                                               innov_hash_map &recurrent_node_innovation_map, int &innovation_front,
                                               int &nodeID_front, std::normal_distribution<float> &distribution,
                                               std::default_random_engine &gen, FastRandIntGenerator &intGenerator,
                                               Parameter &param) : Individual(inputNum, outputNum, innovation_map,
                                                                              node_innovation_set,
                                                                              recurrent_innovation_map,
                                                                              recurrent_node_innovation_map,
                                                                              innovation_front, nodeID_front,
                                                                              distribution, gen, intGenerator, param) {

}

void Individual::save(const char *filename, Individual * ind) {
    std::ofstream out(filename, std::ios_base::binary);
    if (out.good()) {
        const std::vector<ConnectionGene>& genome = ind->connGenome;
        unsigned size = genome.size();
        out.write((char*) &size, sizeof(unsigned));

        out.write((char*) &ind->input_num, sizeof(unsigned));
        out.write((char*) &ind->output_num, sizeof(unsigned));
        out.write((char*) &ind->hidden_num, sizeof(unsigned));
        out.write((char*) &ind->param.actFunc, sizeof(ActivationFunction));

        for (const ConnectionGene& g : genome) {
            out.write((char*) &g.in, sizeof(int));
            out.write((char*) &g.out, sizeof(int));
            out.write((char*) &g.innov, sizeof(int));
            out.write((char*) &g.weight, sizeof(float));
            out.write((char*) &g.is_recurrent, sizeof(bool));
            out.write((char*) &g.enabled, sizeof(bool));
        }
        out.close();
    } else {
        printf("Cannot open file %s for writing", filename);
    }
}

Individual* Individual::load(const char *filename) {
    std::ifstream in(filename,std::ios_base::binary);
    if(in.good()) {
        unsigned size;
        in.read((char *)&size,sizeof(unsigned));
        std::vector<ConnectionGene> genome(size);
        unsigned inputNum;
        unsigned outputNum;
        unsigned hiddenNum;
        ActivationFunction actFunc;

        in.read((char*) &inputNum, sizeof(unsigned));
        in.read((char*) &outputNum, sizeof(unsigned));
        in.read((char*) &hiddenNum, sizeof(unsigned));
        in.read((char*) &actFunc, sizeof(ActivationFunction));


        for (ConnectionGene& g : genome) {
            in.read((char*) &g.in, sizeof(int));
            in.read((char*) &g.out, sizeof(int));
            in.read((char*) &g.innov, sizeof(int));
            in.read((char*) &g.weight, sizeof(float));
            in.read((char*) &g.is_recurrent, sizeof(bool));
            in.read((char*) &g.enabled, sizeof(bool));
        }
        in.close();

        innov_hash_map void_map;
        int void_int;
        std::normal_distribution<float> void_dist;
        std::default_random_engine void_gen;
        FastRandIntGenerator void_intGen;
        Parameter void_param;
        void_param.actFunc = actFunc;

        Individual* ind =  new Individual(genome, inputNum, outputNum, void_map, void_map, void_map, void_map, void_int, void_int,
                              void_dist, void_gen, void_intGen, void_param);
//        ind->print_genes();
        ind->hidden_num = hiddenNum;
        ind->buildPhenotype();
        return ind;
    } else {
        printf("Cannot open file %s for reading", filename);
        return nullptr;
    }
}

//void NonRecurrentIndividual::mutate_disable_genes(const float disable_rate) {
//    assert(phenotype != nullptr);
//    unsigned genome_size = connGenome.size();
//    binomial_distribution<int> binom(genome_size, disable_rate);
//
//    int numToMutate = binom(generator);
//
//    for (unsigned i = 0; i < numToMutate; i++) {
//        ConnectionGene &g = connGenome[intGen.xorshf96() % genome_size];
//        if (g.enabled) {
//            g.enabled = false;
//            phenotype->disable(g.in, g.out);
//
//        } else {
//            phenotype->updateWeight(g, actFunc);
//            if (phenotype->testBackwardRecurrency(g.out)) {
//                phenotype->disable(g.in, g.out);
//            } else {
//                g.enabled = true;
//            }
//        }
//    }
//}
