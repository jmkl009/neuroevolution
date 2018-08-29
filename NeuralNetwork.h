//
// Created by WangJingjin on 2018/7/19.
//

#ifndef SRC_NEURALNETWORK_H
#define SRC_NEURALNETWORK_H


//
// Created by WangJingjin on 2018/7/17.
//

#include <vector>
#include "Gene.h"
#include "Utils.h"
#include "Innovation.h"
#include <sparsehash/dense_hash_map>

using namespace google;

enum ActivationFunction {Sigmoid = 0, Tanh, Relu, Binary, Rough_Sigmoid, Steepened_Sigmoid, Gaussian};

inline float _relu(float in) {
    return in >= 0 ? in : 0;
}

inline float _binary(float in) {
    return in >= 0;
}

inline float _sigmoid(float in) {
    return 1.0/(1.0 + expf(-in));
}

inline float _steepened_sigmoid(float in) {
    return 1.0/(1.0 + expf(-4.9*in));
}

inline float _rough_sigmoid(float in) {
    return in / (1.0 + fabsf(in));
}

inline float _gaussian(float in) {
    return expf(-(in * in));
}

//TODO: Implement to use more than only a single activation function
//TODO: Add depth into attribute to encourage shallower networks.
class NeuralNetwork {


public:
    class Neuron {

    protected:
        class Link {
        public:
            Neuron* input;
            float weight;
            Link(Neuron* in, float w) : input(in), weight(w){};

            virtual float getSignal() {
                return input->activation * weight;
            }
            bool operator==(const Neuron* n) const{
                return input == n;
            }
        };

        class RecurrentLink : public Link {
        public:
            RecurrentLink(Neuron* in, float w) : Link(in, w){};
            virtual float getSignal() {
                return input->prev_activation * weight;
            }
        };

    public:
        typedef std::vector<Neuron*> neuron_list;
        typedef std::vector<Link> connection_list;
        typedef std::vector<RecurrentLink> recurrent_connection_list;

        float activation;
        float prev_activation;
//        bool activated;
        //dense_hash_map<int, Neuron*> outgoings;
        neuron_list outgoings;
        unsigned depth;
        bool depth_is_valid;

    public:
        const int id;
        explicit Neuron(int nID);
        virtual ~Neuron() = default;
        virtual void print_info();

        virtual void disconnectOutGoing(int nID);
        virtual unsigned resolveDepth() = 0;
        virtual void invalidateDepth() = 0;
        virtual void updateDepth(unsigned newDepth) = 0;
        virtual float activate() = 0;
        virtual void addOutGoingNeuron(Neuron* outgoing);
    };

    class InputNeuron : public Neuron {
    public:
        void setInput(const float input);

        explicit InputNeuron(int nID);
        virtual unsigned resolveDepth();
        virtual void invalidateDepth();
        virtual void updateDepth(unsigned newDepth);

        virtual float activate();

        virtual void print_info();

        virtual ~InputNeuron() = default;
    };

    class OutputNeuron : public Neuron {
    public:

        connection_list inputs;
        recurrent_connection_list recurrent_inputs;

//        const neuron_list &getInputNeurons() const;
        std::vector<int> getInputNIDs() const;

        virtual void addOutGoingNeuron(Neuron* outgoing);

    protected:
        virtual void invalidateDepth();
        virtual void updateDepth(unsigned newDepth);
        virtual unsigned resolveDepth();

        virtual void print_info();

    public:
        explicit OutputNeuron(int nID);
        void addInputNeuron(Neuron* neuron, const float weight);
        void addRecurrentInputNeuron(Neuron* neuron, const float weight);
        bool isLinkable(const unsigned inputNIdUpperBound, const unsigned maxLinkableNeuronNums) const;
        bool isRecurrentLinkable(const unsigned inputNIdUpperBound, const unsigned maxLinkableNeuronNums) const;
        bool isConnectedTo(int nID) const;
        bool isConnectedTo(Neuron *n, bool is_recurrent) const;

        void disconnect(int nID, bool is_recurrent);
        void updateWeight(int nID, float weight, const bool is_recurrent);

        virtual float activate();

        virtual ~OutputNeuron() = default;
    };

    class HiddenNeuron : public OutputNeuron {
    private:
        ActivationFunction actFunc;
    public:
        float activation_sum;

    protected:
        virtual unsigned resolveDepth();
        //virtual void invalidateDepth();
        virtual void disconnectOutGoing(int nID);
        virtual void addOutGoingNeuron(Neuron* outgoing);

    public:
        HiddenNeuron(int nID, ActivationFunction act);

        virtual float activate();
        virtual void print_info();

        virtual ~HiddenNeuron() = default;
    };

private:
    std::vector<Neuron*> neurons;
    dense_hash_map<int, int> id_to_idx;
    std::vector<std::vector<Neuron*> > neurons_by_depth;

private:
    bool neurons_by_depth_is_valid;

public:
    const unsigned inputIdUpperBound;
    const unsigned outputIdUpperBound;
    unsigned max_depth;
    bool max_depth_is_valid;

    const std::vector<Neuron*> &getNeuronVec() const;

private:

    //Disable it for cases where there is only a single output for faster evaluation.
    // bool enable_BFS;

    Neuron* findAndInsertNeuron(int id, ActivationFunction actFunc);


//TODO: determine whether to propagate BFS or DFS, or implement them respetively
//TODO: add check for recurrent neurons.
public:
    NeuralNetwork(std::vector<ConnectionGene> &connGenome, unsigned int inputNum, unsigned int outputNum,
                  ActivationFunction actFunc, unsigned int hiddenNum = 0);
    // NeuralNetwork(const NeuralNetwork& other) = default;
    NeuralNetwork(NeuralNetwork&& other) noexcept;

    ~NeuralNetwork();
    //TODO: abandon the DFS implementation, instead we use the BFS implementation.
    std::vector<float> evaluate(const std::vector<float>& inputs);
    void tick(bool identity_recurrent_activation);

    //If outputIndependence is true, then the output neurons cannot be linked with each other.
    //If outputSelfIndependence is true, then the output neurons cannot be linked with itself.
    std::vector<Neuron*> getLinkableNeurons() const;
    std::vector<Neuron*> getRecurrentLinkableNeurons() const;
    void update(const ConnectionGene& newGene, ActivationFunction actFunc);
    void update(const std::vector<ConnectionGene>& newGenes, ActivationFunction actFunc);
    void updateWeight(const int in, const int out, const float weight, const bool is_recurrent);
    void disable(const ConnectionGene& disabledGene);
    //TODO: implement print_info()
    void print_info() const;

    Neuron* getNeuronById(int id) const;
    Neuron* getNeuronByIdx(int id) const;
    int getIdxById(int id) const;

    unsigned getNetworkSize() const;

    unsigned getMaxDepth();
    std::vector<std::vector<unsigned> > getIdsByDepth();

    const std::vector<std::vector<Neuron *>> &getNeuronsByDepth();

    bool testBackwardRecurrency(int in_node_id, int out_node_id);

    unsigned getHiddenNum() const;

    void flush();

    // static void compare(NeuralNetwork* net1, NeuralNetwork* net2);

  //  void draw(const std::vector<ConnectionGene>& genotype, unsigned width = 600, unsigned height = 300, bool horizontal = true);
private:
    void resolveDepth();
    void resolveNeuronsByDepth();

};



#endif //SRC_PRACTICE_NEURALNETWORK_H
