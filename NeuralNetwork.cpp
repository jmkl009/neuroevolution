//
// Created by WangJingjin on 2018/7/19.
//

#include "NeuralNetwork.h"

//
// Created by WangJingjin on 2018/7/17.
//
//#include <queue>
#include <cmath>



//NeuralNetwork::Neuron::Neuron(int nID) : id(nID), activation(0.0), activated(false), outgoings(), depth(0), depth_is_valid(false) {
//    //printf("A neuron is constructed.\n");
////    outgoings.set_empty_key(-1);
////    outgoings.set_deleted_key(-2);
////    outgoings.max_load_factor(0.7);
//    outgoings.reserve(8);
//};

NeuralNetwork::Neuron::Neuron(int nID) : id(nID), activation(0.0), prev_activation(0.0), outgoings(), depth(0), depth_is_valid(false) {
    //printf("A neuron is constructed.\n");
//    outgoings.set_empty_key(-1);
//    outgoings.set_deleted_key(-2);
//    outgoings.max_load_factor(0.7);
//    outgoings.reserve(8);
};

void NeuralNetwork::Neuron::print_info() {
    printf("ID: %d\n", id);
    printf("Activation: %f\n", activation);
    printf("Prev activation: %f\n", prev_activation);
    printf("At depth: %d\n", this->resolveDepth());
    printf("Outgoing neuron ids: ");

    for (Neuron*& n : outgoings) {
        printf("%d, ", n->id);
    }
    printf("\n");



//    if(activated) {
//        printf("activated: true\n\n");
//    } else {
//        printf("activated: false\n\n");
//    }
}

NeuralNetwork::InputNeuron::InputNeuron(int nID) : Neuron(nID) {
}


void NeuralNetwork::Neuron::addOutGoingNeuron(NeuralNetwork::Neuron *outgoing) {
    //outgoings.emplace(outgoing->id, outgoing);
    outgoings.emplace_back(outgoing);
    unsigned potential_new_depth = outgoing->resolveDepth() + 1;
    if (potential_new_depth >= depth) {
        depth_is_valid = false;
    }
}

void NeuralNetwork::Neuron::disconnectOutGoing(int nID) {
//    unsigned outgoing_size = outgoings.size();
    for (auto iter = outgoings.begin(); iter != outgoings.end(); ++iter) {
        if ((*iter)->id == nID) {
//            if (outgoings[i]->resolveDepth() + 1 == depth) {
//                depth_is_valid = false;
//            }
            outgoings.erase(iter);
            break;
        }
    }
    // outgoings.erase(outgoings.find(nID));
    depth_is_valid = false;
}

//void NeuralNetwork::Neuron::flush() {
//    if (activated) {
//        activated = false;
////        for (auto iter = outgoings.begin(); iter != outgoings.end(); ++iter) {
////            iter->second->flush();
////        }
//        for (Neuron*& n : outgoings) {
//            n->flush();
//        }
//    }
//}

unsigned NeuralNetwork::InputNeuron::resolveDepth() {
    if (!depth_is_valid) {
//        for (auto iter = outgoings.begin(); iter != outgoings.end(); ++iter) {
//            unsigned temp_depth = iter->second->resolveDepth() + 1;
//            if (temp_depth > depth) {
//                depth = temp_depth;
//            }
//        }
        for (Neuron*& n : outgoings) {
            unsigned temp_depth = n->resolveDepth() + 1;
            if (temp_depth > depth) {
                depth = temp_depth;
            }
        }
        depth_is_valid = true;
    }
    return depth;
}

void NeuralNetwork::InputNeuron::invalidateDepth() {
    depth_is_valid = false;
}

void NeuralNetwork::InputNeuron::setInput(const float input) {
    activation = input;
}

float NeuralNetwork::InputNeuron::activate() {
    //print_info();
    return activation;
}

void NeuralNetwork::InputNeuron::updateDepth(unsigned newDepth) {
    if (newDepth > depth) {
        depth_is_valid = false;
    }
}

void NeuralNetwork::InputNeuron::print_info() {
    printf("This is an input neuron\n");
    Neuron::print_info();
}


NeuralNetwork::OutputNeuron::OutputNeuron(int nID) : Neuron(nID), inputs(), recurrent_inputs() {
//    input_neurons.reserve(8);
//    weights.reserve(8);
}

void NeuralNetwork::OutputNeuron::addInputNeuron(NeuralNetwork::Neuron *neuron, const float weight) {
    inputs.emplace_back(Link(neuron, weight));
}

float NeuralNetwork::OutputNeuron::activate() {
    //   assert(input_neurons.size() == weights.size());
//
//    if (activated) {
//        return activation;
//    }
//    activated = true;
//
    float activation_sum = 0.0;
//    size_t input_size = input_neurons.size();
//
//    for (int i = 0; i < input_size; i++) {
//        activation_sum += input_neurons[i]->activate() * weights[i];
//    }
//    activation = activation_sum;
    for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
        activation_sum += iter->getSignal();
    }

//    size_t recurrent_input_size = recurrent_input_neurons.size();
    for (auto iter = recurrent_inputs.begin(); iter != recurrent_inputs.end(); ++iter) {
        activation_sum += iter->getSignal();
    }

    activation = activation_sum;

    return activation;
}

bool NeuralNetwork::OutputNeuron::isLinkable(const unsigned inputNIdUpperBound, const unsigned maxLinkableNeuronNums) const {

    if (inputs.size() < maxLinkableNeuronNums) {
        return true;
    }

    unsigned count = 0;
    for (Link const &g : inputs) {
        if (g.input->id >= inputNIdUpperBound) {
            ++count;
        }
    }

    return count < maxLinkableNeuronNums;
}

bool NeuralNetwork::OutputNeuron::isRecurrentLinkable(const unsigned inputNIdUpperBound,
                                                      const unsigned maxLinkableNeuronNums) const {
    if (recurrent_inputs.size() < maxLinkableNeuronNums) {
        return true;
    }

    unsigned count = 0;
    for (RecurrentLink const &g : recurrent_inputs) {
        if (g.input->id >= inputNIdUpperBound) {
            ++count;
        }
    }

    return count < maxLinkableNeuronNums;
}

std::vector<int> NeuralNetwork::OutputNeuron::getInputNIDs() const {
    std::vector<int> nIDs;
    nIDs.reserve(inputs.size());
    for (Link const &n : inputs) {
        nIDs.emplace_back(n.input->id);
    }
    return nIDs;
}

void NeuralNetwork::OutputNeuron::disconnect(int nID, bool is_recurrent) {
    if (!is_recurrent) {
//        printf("disconnect called\n");
        for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
            Link& n = *iter;
            if (n.input->id == nID) {
                n.input->disconnectOutGoing(id);
                inputs.erase(iter);
                return;
            }
        }
    } else {
        for (auto iter = recurrent_inputs.begin(); iter != recurrent_inputs.end(); ++iter) {
            RecurrentLink& n = *iter;
            if (n.input->id == nID) {
                recurrent_inputs.erase(iter);
                return;
            }
        }
    }

}

void NeuralNetwork::OutputNeuron::updateWeight(int nID, float weight, const bool is_recurrent) {
    if (!is_recurrent) {
//        unsigned input_size = input_neurons.size();

        for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
            if ((*iter).input->id == nID) {
                (*iter).weight = weight;
                return;
            }
        }
    } else {
//        unsigned recurrent_input_size = recurrent_input_neurons.size();
        for (auto iter = recurrent_inputs.begin(); iter != recurrent_inputs.end(); ++iter) {
            if ((*iter).input->id == nID) {
                (*iter).weight = weight;
                return;
            }
        }
    }
}

bool NeuralNetwork::OutputNeuron::isConnectedTo(int nID) const {
    for (Link const &in : inputs) {
        if (in.input->id == nID) {
            return true;
        }
    }

    for (RecurrentLink const &in : recurrent_inputs) {
        if (in.input->id == nID) {
            return true;
        }
    }
    return false;
}

bool NeuralNetwork::OutputNeuron::isConnectedTo(Neuron *n, bool is_recurrent) const {
    if (!is_recurrent) {
        return find(inputs.begin(), inputs.end(), n) != inputs.end();
    }
    return find(recurrent_inputs.begin(), recurrent_inputs.end(), n) != recurrent_inputs.end();
}

void NeuralNetwork::OutputNeuron::addOutGoingNeuron(NeuralNetwork::Neuron *outgoing) {
//    outgoings.emplace(outgoing->id, outgoing);
    outgoings.emplace_back(outgoing);
}

void NeuralNetwork::OutputNeuron::invalidateDepth() {
    for (Link& n : inputs) {
        n.input->invalidateDepth();
    }
    depth_is_valid = false;
}

unsigned NeuralNetwork::OutputNeuron::resolveDepth() {
    return 0;
}

void NeuralNetwork::OutputNeuron::updateDepth(unsigned newDepth) {
    if (newDepth > depth) {
        for (Link& n : inputs) {
            n.input->updateDepth(newDepth + 1);
        }
        depth_is_valid = false;
    }
}

void NeuralNetwork::OutputNeuron::addRecurrentInputNeuron(NeuralNetwork::Neuron *neuron, const float weight) {
    recurrent_inputs.emplace_back(RecurrentLink(neuron, weight));
}

void NeuralNetwork::OutputNeuron::print_info() {
    printf("This is an output neuron");
    Neuron::print_info();

    printf("Input neuron ids and weights: ");
    for (Link& l : inputs) {
        printf("%d | %f,  ", l.input->id, l.weight);
    }
    printf("\n");

    printf("Recurrent Input neuron ids: ");
    for (RecurrentLink& l : recurrent_inputs) {
        printf("%d | %f,  ", l.input->id, l.weight);
    }
    printf("\n");
}


NeuralNetwork::HiddenNeuron::HiddenNeuron(int nID, ActivationFunction act) : OutputNeuron(nID), actFunc(act) {}

float NeuralNetwork::HiddenNeuron::activate() {

//    if (activated) {
//        return activation;
//    }
//    activated = true;

    activation_sum = 0.0;
//    size_t input_size = input_neurons.size();
//    for (int i = 0; i < input_size; i++) {
//        activation_sum += input_neurons[i]->activate() * weights[i];
//    }
    for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
        activation_sum += iter->getSignal();
    }

//    size_t recurrent_input_size = recurrent_input_neurons.size();
    for (auto iter = recurrent_inputs.begin(); iter != recurrent_inputs.end(); ++iter) {
        activation_sum += iter->getSignal();
    }

    switch(actFunc) {
        case Sigmoid:
            activation = _sigmoid(activation_sum);
            break;
        case Tanh:
            activation = tanhf(activation_sum);
            break;
        case Relu:
            activation = _relu(activation_sum);
            break;
        case Binary:
            activation = _binary(activation_sum);
            break;
        case Rough_Sigmoid:
            activation = _rough_sigmoid(activation_sum);
            break;
        case Steepened_Sigmoid:
            activation = _steepened_sigmoid(activation_sum);
            break;
        case Gaussian:
            activation = _gaussian(activation_sum);
            break;
        default:
            break;
    }
    //print_info();
    return activation;
}

void NeuralNetwork::HiddenNeuron::disconnectOutGoing(int nID) {
//    unsigned outgoing_size = outgoings.size();

    for (auto iter = outgoings.begin(); iter != outgoings.end(); ++iter) {
        Neuron*& n = *iter;
        if (n->id == nID) {
            if (depth_is_valid && n->resolveDepth() + 1 >= depth) {
                invalidateDepth();
            }
            outgoings.erase(iter);
            break;
        }
    }
}

unsigned NeuralNetwork::HiddenNeuron::resolveDepth() {
    if (!depth_is_valid) {
        depth = 0;
        for (Neuron*& n : outgoings) {
            unsigned temp_depth = n->resolveDepth() + 1;
            if (temp_depth > depth) {
                depth = temp_depth;
            }
        }
        depth_is_valid = true;
    }
    return depth;
}

void NeuralNetwork::HiddenNeuron::addOutGoingNeuron(NeuralNetwork::Neuron *outgoing) {
    outgoings.emplace_back(outgoing);
    unsigned potential_new_depth = outgoing->resolveDepth() + 1;
    if (depth_is_valid && potential_new_depth >= depth) {
        invalidateDepth();
    }
}

void NeuralNetwork::HiddenNeuron::print_info() {
    printf("This is a hidden neuron");
    Neuron::print_info();

    printf("Input neuron ids and weights: ");
    for (Link& l : inputs) {
        printf("%d | %f,  ", l.input->id, l.weight);
    }
    printf("\n");

    printf("Recurrent Input neuron ids: ");
    for (RecurrentLink& l : recurrent_inputs) {
        printf("%d | %f,  ", l.input->id, l.weight);
    }
    printf("\n");
}


NeuralNetwork::Neuron* NeuralNetwork::findAndInsertNeuron(int id, ActivationFunction actFunc) {

    if (id >= outputIdUpperBound) {

        auto search_result = id_to_idx.find(id);
        if (search_result == id_to_idx.end()) {

            id_to_idx.emplace(id, (int)neurons.size());
            neurons.emplace_back(new HiddenNeuron(id, actFunc));
            return neurons.back();
        }

        return neurons[search_result->second];

    }
    return neurons[id];
}


NeuralNetwork::NeuralNetwork(std::vector<ConnectionGene> &connGenome, unsigned int inputNum, unsigned int outputNum,
                             ActivationFunction actFunc, unsigned int hiddenNum) : neurons(inputNum + outputNum), id_to_idx(),
                                                                                   inputIdUpperBound(inputNum),
                                                                                   outputIdUpperBound(inputNum + outputNum),
                                                                                   neurons_by_depth(), neurons_by_depth_is_valid(false){

    neurons.reserve(inputNum + outputNum + hiddenNum + 5);
    id_to_idx.set_empty_key(-1);
    id_to_idx.resize(neurons.size() * 2);
    id_to_idx.max_load_factor(0.7);

    for (unsigned i = 0; i < inputIdUpperBound; i++) {
        neurons[i] = new InputNeuron(i);
    }

    for (unsigned i = inputIdUpperBound; i < outputIdUpperBound; i++) {
        neurons[i] = new OutputNeuron(i);
    }

//    neurons.resize(neurons_size);
    for (ConnectionGene& g : connGenome) {
        if (g.enabled) {
            Neuron* outNeuron = findAndInsertNeuron(g.out, actFunc);
            Neuron* inNeuron = findAndInsertNeuron(g.in, actFunc);
            if (!g.is_recurrent) {
                ((OutputNeuron *)outNeuron)->addInputNeuron(inNeuron, g.weight);
                inNeuron->addOutGoingNeuron(((OutputNeuron *)outNeuron));
            } else {
                ((OutputNeuron *)outNeuron)->addRecurrentInputNeuron(inNeuron, g.weight);
            }
        }
    }

    max_depth = 0;
    max_depth_is_valid = false;

}


//TODO:modify this function
std::vector<float> NeuralNetwork::evaluate(const std::vector<float>& inputs) {
    assert(inputs.size() == inputIdUpperBound);

    if(!neurons_by_depth_is_valid) {
        resolveNeuronsByDepth();
    }

    std::vector<float> output = std::vector<float>();
    output.reserve(outputIdUpperBound - inputIdUpperBound);

    for (int i = 0; i < inputIdUpperBound; i++) {
        ((InputNeuron *) neurons[i])->setInput(inputs[i]);
    }

    if (max_depth > 0) {
        for (unsigned i = max_depth - 1; i > 0; i--) {
            for (Neuron*& n : neurons_by_depth[i]) {
                n->activate();
            }
        }
    }

    for (int i = inputIdUpperBound; i < outputIdUpperBound; i++) {
        output.emplace_back(((OutputNeuron *) neurons[i])->activate());
    }

//    for (int i = 0; i < inputIdUpperBound; i++) {
//        (neurons[i])->flush();
//    }

    return output;
}

void NeuralNetwork::tick(bool identity_recurrent_activation) {
    unsigned network_size = neurons.size();
    for (unsigned i = 0; i < outputIdUpperBound; ++i) {
        neurons[i]->prev_activation = neurons[i]->activation;
    }
    if (identity_recurrent_activation) {
        for (unsigned i = outputIdUpperBound; i < network_size; ++i) {
            neurons[i]->prev_activation = ((HiddenNeuron*)neurons[i])->activation_sum;
        }
    } else {
        for (unsigned i = outputIdUpperBound; i < network_size; ++i) {
            neurons[i]->prev_activation = neurons[i]->activation;
        }
    }
}



NeuralNetwork::~NeuralNetwork() {
    for (Neuron *& n : neurons) {
        delete n;
    }
}

void NeuralNetwork::print_info() const{
    printf("number of input neurons: %d\n", inputIdUpperBound);
    printf("number of hidden neurons: %lu\n", neurons.size() - outputIdUpperBound);
    printf("number of output neurons: %d\n", outputIdUpperBound - inputIdUpperBound);
}

std::vector<NeuralNetwork::Neuron*> NeuralNetwork::getLinkableNeurons() const{

    std::vector<NeuralNetwork::Neuron*> linkableNeurons;
    unsigned neurons_size = neurons.size();
    unsigned maxLinkableNeuronNums = neurons_size - inputIdUpperBound;

    linkableNeurons.reserve(maxLinkableNeuronNums);
    for (int i = inputIdUpperBound; i < neurons_size; i++) {
        OutputNeuron * const&n = (OutputNeuron*)neurons[i];
        if (n->isLinkable(inputIdUpperBound, maxLinkableNeuronNums)) {
            linkableNeurons.emplace_back(n);
        }
    }

    return linkableNeurons;
}

std::vector<NeuralNetwork::Neuron *> NeuralNetwork::getRecurrentLinkableNeurons() const {
    std::vector<NeuralNetwork::Neuron*> recurrentLinkableNeurons;
    unsigned neurons_size = neurons.size();
    unsigned maxLinkableNeuronNums = neurons_size - inputIdUpperBound;

    recurrentLinkableNeurons.reserve(maxLinkableNeuronNums);
    for (int i = inputIdUpperBound; i < neurons_size; i++) {
        OutputNeuron * const&n = (OutputNeuron*)neurons[i];
        if (n->isRecurrentLinkable(inputIdUpperBound, maxLinkableNeuronNums)) {
            recurrentLinkableNeurons.emplace_back(n);
        }
    }

    return recurrentLinkableNeurons;
}

const std::vector<NeuralNetwork::Neuron *> &NeuralNetwork::getNeuronVec() const {
    return neurons;
}

void NeuralNetwork::update(const ConnectionGene &newGene, ActivationFunction actFunc) {
    Neuron* outNeuron = findAndInsertNeuron(newGene.out, actFunc);
    Neuron* inNeuron = findAndInsertNeuron(newGene.in, actFunc);

    if (!newGene.is_recurrent) {
        ((OutputNeuron *)outNeuron)->addInputNeuron(inNeuron, newGene.weight);
        inNeuron->addOutGoingNeuron(((OutputNeuron *)outNeuron));
        max_depth_is_valid = false;
        neurons_by_depth_is_valid = false;
    } else {
        ((OutputNeuron *)outNeuron)->addRecurrentInputNeuron(inNeuron, newGene.weight);
    }
}

void NeuralNetwork::updateWeight(const int in, const int out, const float weight, const bool is_recurrent) {
    ((OutputNeuron *) getNeuronById(out))->updateWeight(in, weight, is_recurrent);
}

void NeuralNetwork::update(const std::vector<ConnectionGene> &newGenes, ActivationFunction actFunc) {
    for (const ConnectionGene &g : newGenes) {
        update(g, actFunc);
    }
}

void NeuralNetwork::disable(const ConnectionGene &disabledGene) {
    if (disabledGene.out < outputIdUpperBound) {
        ((OutputNeuron *) neurons[disabledGene.out])->disconnect(disabledGene.in, disabledGene.is_recurrent);
        // (neurons[disabledGene.in])->disconnectOutGoing(disabledGene.out);
    } else {
        auto search_result = id_to_idx.find(disabledGene.out);
        if (search_result != id_to_idx.end()) {
            ((OutputNeuron *) neurons[search_result->second])->disconnect(disabledGene.in, disabledGene.is_recurrent);
        }
//        auto search_result_in = id_to_idx.find(disabledGene.in);
//        if (search_result_in != id_to_idx.end()) {
//            (neurons[search_result_in->second])->disconnectOutGoing(disabledGene.out);
//        }
    }
    if (!disabledGene.is_recurrent) {
        max_depth_is_valid = false;
        neurons_by_depth_is_valid = false;
    }
}
//
//void NeuralNetwork::disable(int in, int out) {
//    if (out < outputIdUpperBound) {
//        ((OutputNeuron *) neurons[out])->disconnect(in);
//       // (neurons[in])->disconnectOutGoing(out);
//    } else {
//        auto search_result = id_to_idx.find(out);
//        if (search_result != id_to_idx.end()) {
//            ((OutputNeuron *) neurons[search_result->second])->disconnect(in);
//        }
////        auto search_result_in = id_to_idx.find(in);
////        if (search_result_in != id_to_idx.end()) {
////            (neurons[search_result_in->second])->disconnectOutGoing(out);
////        }
//    }
//    max_depth_is_valid = false;
//    neurons_by_depth_is_valid = false;
//}

NeuralNetwork::Neuron *NeuralNetwork::getNeuronById(int id) const {
    if (id < outputIdUpperBound) {
        return neurons[id];
    }

    auto search_result = id_to_idx.find(id);
    if (search_result != id_to_idx.end()) {
        return neurons[search_result->second];
    }

    return nullptr;
}

NeuralNetwork::NeuralNetwork(NeuralNetwork &&other) noexcept
        : neurons(std::move(other.neurons)), id_to_idx(std::move(other.id_to_idx)), inputIdUpperBound(other.inputIdUpperBound)
        , outputIdUpperBound(other.outputIdUpperBound), neurons_by_depth(std::move(other.neurons_by_depth)), neurons_by_depth_is_valid(other.neurons_by_depth_is_valid) {

    max_depth = other.max_depth;
    max_depth_is_valid = other.max_depth_is_valid;
    other.neurons = std::vector<Neuron*>();
    other.id_to_idx = dense_hash_map<int, int>();
    other.id_to_idx.max_load_factor(0.7);

    //   printf("NeuralNetwork move constructor called");
}

unsigned NeuralNetwork::getNetworkSize() const {
    return neurons.size();
}

int NeuralNetwork::getIdxById(int id) const {
    if (id < outputIdUpperBound) {
        return id;
    }
    auto search_result = id_to_idx.find(id);
    if (search_result != id_to_idx.end()) {
        return search_result->second;
    }

    return -1;
}

NeuralNetwork::Neuron *NeuralNetwork::getNeuronByIdx(int id) const {
    return neurons[id];
}

unsigned NeuralNetwork::getMaxDepth() {
    if (!max_depth_is_valid) {
        resolveDepth();
    }
    return max_depth;
}

std::vector<std::vector<unsigned>> NeuralNetwork::getIdsByDepth() {

    if (!max_depth_is_valid) {
        resolveDepth();
    }
    std::vector<std::vector<unsigned>> IdsByDepth(max_depth + 1);

    for (unsigned i = 0; i < inputIdUpperBound; i++) {
        IdsByDepth[0].emplace_back(neurons[i]->id);
    }

    unsigned neurons_size = neurons.size();

    for (unsigned i = inputIdUpperBound; i < neurons_size; i++) {
        IdsByDepth[neurons[i]->resolveDepth()].emplace_back(neurons[i]->id);
    }
//
//    for (Neuron*& n : neurons) {
//        IdsByDepth[n->resolveDepth()].emplace_back(n->id);
//    }

    return IdsByDepth;
}

void NeuralNetwork::resolveDepth() {
    max_depth = 0;
    for (unsigned i = 0; i < inputIdUpperBound; i++) {
        unsigned temp_depth = neurons[i]->resolveDepth();
        if (max_depth < temp_depth) {
            max_depth = temp_depth;
        }
    }
    max_depth_is_valid = true;
}

void NeuralNetwork::resolveNeuronsByDepth() {
    if (!max_depth_is_valid) {
        resolveDepth();

        neurons_by_depth = std::vector<std::vector<Neuron *>>(max_depth + 1);

        for (unsigned i = 0; i < inputIdUpperBound; i++) {
            neurons_by_depth[max_depth].emplace_back(neurons[i]);
        }

        unsigned neurons_size = neurons.size();

        for (unsigned i = inputIdUpperBound; i < neurons_size; i++) {
            unsigned resolved_depth = neurons[i]->resolveDepth();
            if (resolved_depth <= max_depth) {
                neurons_by_depth[resolved_depth].emplace_back(neurons[i]);
            }
        }
    }

    neurons_by_depth_is_valid = true;
}

const std::vector<std::vector<NeuralNetwork::Neuron *> > &NeuralNetwork::getNeuronsByDepth() {
    if (!neurons_by_depth_is_valid) {
        resolveNeuronsByDepth();
    }
    return neurons_by_depth;
}

bool NeuralNetwork::testBackwardRecurrency(int in_node_id, int out_node_id) {
    if (in_node_id == out_node_id) {
        return true;
    }

    if (in_node_id < inputIdUpperBound) {
        return false;
    }

    Neuron* in_node = getNeuronById(in_node_id);
    if (in_node == nullptr) { //Not found the node so the node can be placed anywhere.
        return false;
    }
    Neuron* out_node = getNeuronById(out_node_id);
    if(out_node == nullptr) {
        return false;
    }

    return in_node->resolveDepth() < out_node->resolveDepth();
}

void NeuralNetwork::flush() {
    for (Neuron*& n : neurons) {
        n->activation = 0.0;
        n->prev_activation = 0.0;
    }
}

unsigned NeuralNetwork::getHiddenNum() const {
    return neurons.size() - outputIdUpperBound;
}
//
// void NeuralNetwork::compare(NeuralNetwork *net1, NeuralNetwork *net2) {
//     const std::vector<Neuron*>& neurons1 = net1->neurons;
//     const std::vector<Neuron*>& neurons2 = net2->neurons;
//
//     unsigned neurons1_size = neurons1.size();
//     unsigned neurons2_size = neurons2.size();
//     printf("net1 size: %lu\n", neurons1_size);
//     printf("net2 size: %lu\n", neurons2_size);
//
// //    unsigned neuron_size_upper = neurons1_size;
// //    if (neurons1_size < neurons2_size) {
// //        printf("net1 is smaller.\n");
// //    } else if (neurons1_size > neurons2_size) {
// //        printf("net2 is smaller.\n");
// //        neuron_size_upper = neurons2_size;
// //    }
//
//     for (unsigned i = net1->inputIdUpperBound; i < net1->outputIdUpperBound; ++i) {
//         if (neurons1[i]->activation != neurons2[i]->activation) {
//             printf("Neuron id: %d activation mismatch.\n", i);
//             printf("Net 1 neuron's info: \n");
//             neurons1[i]->print_info();
//             printf("Net 2 neuron's info: \n");
//             neurons2[i]->print_info();
//             printf("\n");
//         }
//     }
// }
