//
// Created by WangJingjin on 2018/7/19.
//

#include "Gene.h"
//#include <utility>
//#include "assert.h"

using namespace std;

ConnectionGene::ConnectionGene(int in, int out, int innov, float weight, bool is_recurrent) : in(in), out(out), innov(innov), weight(weight), enabled(true), is_recurrent(is_recurrent) {
    //printf("ConnectionGene innov: %d created through constructing\n", innov);
};

void ConnectionGene::print_info() const {
    printf("in: %d\n", in);
    printf("out: %d\n", out);
    printf("weight: %f\n", weight);

    if (is_recurrent) {
        printf("is recurrent: true\n");
    } else {
        printf("is recurrent: false\n");
    }

    printf("innov: %d\n", innov);

    if (enabled) {
        printf("enable status: enabled\n");
    } else {
        printf("enable status: disabled\n");
    }
}

std::string ConnectionGene::get_info() const {
    std::stringstream str;
    str << "in: " << in << "\n";
    str << "out: " << out << "\n";
    str << "weight: " << weight << "\n";
    str << "is recurrent: " << is_recurrent << "\n";
    str << "innov: " << innov << "\n";
    str << "enable status: " << (enabled ? "enabled" : "disabled") << "\n";
    return str.str();
}

bool ConnectionGene::operator==(const ConnectionGene& other) const{
    return innov == other.innov;
}

bool ConnectionGene::operator<(const ConnectionGene& other) const{
    return innov < other.innov;
}

bool ConnectionGene::operator>(const ConnectionGene &other) const {
    return innov > other.innov;
}

bool ConnectionGene::operator<=(const ConnectionGene &other) const {
    return innov <= other.innov;
}

bool ConnectionGene::operator>=(const ConnectionGene &other) const {
    return innov >= other.innov;
}

ConnectionGene &ConnectionGene::operator=(const ConnectionGene &other) noexcept {
    if (this != &other) {
        in = other.in;
        out = other.out;
        innov = other.innov;
        weight = other.weight;
        is_recurrent = other.is_recurrent;
        enabled = other.enabled;
    }
    return *this;
}

ConnectionGene::ConnectionGene() {

}
