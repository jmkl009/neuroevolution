//
// Created by WangJingjin on 2018/7/16.
//

#ifndef SRC_UTILS_H
#define SRC_UTILS_H

#include <cstdio>
#include <utility>
// #include <unordered_set>
#include <algorithm>
#include <cstdlib>

//extern uint32_t lcg64_temper(uint64_t *seed); // see R.. code
//
//static uint64_t gseed = clock(); //  Initialize this in some fashion.
//
//bool rand_bool(void) {
//    static uint32_t rbits;
//    printf("rbits: %llu\n", rbits);
//    printf("gseed: %llu\n", gseed);
//    if (gcount == 0) {
//        gcount = 31;  // I'd consider using 31 here, just to cope with some LCG weaknesses.
//        rbits = lcg64_temper(&gseed);
//    }
//    gcount--;
//    bool b = rbits & 1;
//    rbits >>= 1;
//    return b;
//}

//struct simple_alloc {
//    static void* Allocate(size_t size){
//        return malloc(size);
//    }
//    static void Free(void* ptr, size_t size){
//        free(ptr);
//    }
//};

namespace mapExt
{
    template<typename myMap>
    std::vector<typename myMap::key_type> Keys(const myMap& m)
    {
        std::vector<typename myMap::key_type> r;
        r.reserve(m.size());
        for (const auto&kvp : m)
        {
            r.emplace_back(kvp.first);
        }
        return r;
    }

    template<typename myMap>
    std::vector<typename myMap::key_type> Keys_sorted(const myMap& m)
    {
        std::vector<typename myMap::key_type> r;
        r.reserve(m.size());
        for (const auto&kvp : m)
        {
            r.emplace_back(kvp.first);
        }
        std::sort(r.begin(), r.end());
        return r;
    }

    // template<typename myMap>
    // std::unordered_set<typename myMap::key_type> Keys_set(const myMap& m)
    // {
    //     std::unordered_set<typename myMap::key_type> r;
    //     r.reserve(m.size() * 1.5);
    //     r.max_load_factor(0.7);
    //     for (const auto&kvp : m)
    //     {
    //         r.emplace(kvp.first);
    //     }
    //     return r;
    // }

    template<typename myMap>
    std::vector<typename myMap::mapped_type> Values(const myMap& m)
    {
        std::vector<typename myMap::mapped_type> r;
        r.reserve(m.size());
        for (const auto&kvp : m)
        {
            r.push_back(kvp.second);
        }
        return r;
    }
}

class FastRandIntGenerator {

private:
    unsigned long x, y, z;
    unsigned gcount;
    uint32_t rbits;

public:
    FastRandIntGenerator() : x(123456789), y(362436069), z(521288629), gcount(31) {
        rbits = xorshf96();
    };

    unsigned long xorshf96() {          //period 2^96-1
        unsigned long t;
        x ^= x << 16;
        x ^= x >> 5;
        x ^= x << 1;

        t = x;
        x = y;
        y = z;
        z = t ^ x ^ y;

        return z;
    };

    bool rand_bool() {
        if (gcount == 0) {
            gcount = 31;
            rbits = xorshf96();
        }
//        printf("gcount: %d\n", gcount);
//        printf("rbits: %d\n", rbits);
        --gcount;
        bool b = rbits & 1;
        rbits >>= 1;
        return b;
    };
};

inline int
pow2roundup (int x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x+1;
}

#endif //SRC_UTILS_H
