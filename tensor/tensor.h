#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include "type.h"


template<typename T>
class tensor{
    public:
      // Constructor
      tensor(std::vector<int> shape, type::backend_type backend);

      // Default Constructor
      tensor();

      // Destructor
      ~tensor();

      // return the size of the tensor
      int64_t get_size() const;

      // return the shape of the tensor
      std::vector<int> get_shape() const;

      // return true if tensor is on GPU
      bool on_gpu();

      // return data given by index, index must be of length size
      T get_data(int index) const;

      // set data given by index, index must be of length size
      void set_data(int index, T data);

      // initialize everything to be 0
      void init_data();

      void set_name(std::string name);

      

    private:
      std::string name = "";
      int64_t size = 0;
      type::backend_type backend = type::backend_type::CPU;
      std::vector<int> shape = {0};
      T* buffer = nullptr;

      // Allocate memory based on backend type
      T* mem_alloc(type::backend_type backend, int64_t size);
};

#include "tensor.cpp"

#endif



