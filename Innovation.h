//
// Created by WangJingjin on 2018/7/16.
//

#ifndef SRC_INNOVATION_H
#define SRC_INNOVATION_H

#include <unordered_map>
#include <List>
#include <google/dense_hash_map>


struct pairhash {
public:
    inline size_t operator()(const std::pair<int, int> &p) const {
        size_t a = p.first;
        size_t sum = a + p.second;
        return ((sum * (sum + 1)) >> 1) + a;
    }
};

class innov_hash_map : public std::unordered_map<std::pair<int, int>, int, pairhash> {};

#endif //SRC_INNOVATION_H
