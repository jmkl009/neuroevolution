//
// Created by WangJingjin on 2018/7/19.
//

#ifndef OPENCV_PRACTICE_GENE_H
#define OPENCV_PRACTICE_GENE_H

#include <cstdio>
#include <sstream>

//enum NodeType {Input, Hidden, Output};

class ConnectionGene {
public:
    int in, out, innov;
    float weight;
    bool is_recurrent;
    bool enabled;

    ConnectionGene(int in, int out, int innov, float weight, bool is_recurrent);

    ConnectionGene();

    void print_info() const;
    std::string get_info() const;
    bool operator==(const ConnectionGene& other) const;
    bool operator<(const ConnectionGene& other) const;
    bool operator>(const ConnectionGene& other) const;
    bool operator<=(const ConnectionGene& other) const;
    bool operator>=(const ConnectionGene& other) const;
    ConnectionGene& operator=(const ConnectionGene& other) noexcept;
};

//class NodeGene {
//public:
//    int id;
//    Activation act;
//    NodeType type;
//
//    NodeGene(int nID, Activation activation, NodeType nodeType);
//    void print_info() const;
//    bool operator==(const NodeGene& other) const;
//};



#endif //OPENCV_PRACTICE_GENE_H
