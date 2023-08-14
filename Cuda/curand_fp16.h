#pragma once

#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>

#include <cstdint>

namespace curand_fp16 {

template <curandRngType_t rng>
struct curand_status_t {
  using type = void;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_MTGP32> {
  using type = curandStateMtgp32_t;
};

template <>
struct curand_status_t<CURAND_RNG_QUASI_SCRAMBLED_SOBOL32> {
  using type = curandStateScrambledSobol32_t;
};

template <>
struct curand_status_t<CURAND_RNG_QUASI_SOBOL32> {
  using type = curandStateSobol32_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_MRG32K3A> {
  using type = curandStateMRG32k3a_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_XORWOW> {
  using type = curandStateXORWOW_t;
};

template <>
struct curand_status_t<CURAND_RNG_PSEUDO_PHILOX4_32_10> {
  using type = curandStatePhilox4_32_10_t;
};

struct generator_t {
  unsigned num_sm;
  unsigned num_threads;
  cudaStream_t cuda_stream;
  curandRngType_t rng_type;
  void* status_ptr;
};

void create(generator_t& gen, const curandRngType_t rng_type);
void destroy(generator_t& gen);
void set_seed(generator_t& gen, const std::uint64_t seed);
void set_cuda_stream(generator_t& gen, cudaStream_t const cuda_stream);

// Uniform distribution
// pm == true  | (-1, 1)
//       false | ( 0, 1)
void uniform(generator_t& gen, half* const ptr, const std::size_t size,
             const bool pm = false);

// Normal distribution
void normal(generator_t& gen, half* const ptr, const std::size_t size,
            const float mean, const float var);
}  // namespace curand_fp16
