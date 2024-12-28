#include "class.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <vector>
#include "type.h"


// Constructor
template<typename T>
tensor<T>::tensor(std::vector<int> shape, type::backend_type backend){
    // Check all elem in shape >= 0
    this->size = 1;
    for(int i = 0; i < shape.size(); i++){
        if(shape[i] < 0){
            std::cout << " Shape must be non-zero!" << std::endl;
            return;
        }
        this->size *= shape[i];
    }

    std::vector<int> shapee(shape);
    this->shape = shapee;
    this->backend = backend;


    // TODO: Implement this wrapper function for memory allocation
    T* buffer = this->mem_alloc(this->backend, this->size);

    this->buffer = buffer;
}

// Default constructor
template<typename T>
tensor<T>::tensor(){
    this->shape = {0};
    this->size = 0;
    this->backend = type::backend_type::CPU;
    this->buffer = nullptr;
}

template<typename T>
int64_t tensor<T>::get_size() const{
    return this->size;
}

template<typename T>     
std::vector<int> tensor<T>::get_shape() const{
    return this->shape;
}

template<typename T>
bool tensor<T>::on_gpu(){
    return this->backend == type::backend_type::GPU;
}

template<typename T>
T tensor<T>::get_data(int index) const{
    assert(index >= 0 && index < this->size);

    return this->buffer[index];
}


template<typename T>      
void tensor<T>::set_data(int index, T data){
    this->buffer[index] = data;
}


// Initialize everything in buffer to be 0      
template<typename T>
void tensor<T>::init_data(){
    for(int i = 0; i < this->size; i++){
        this->buffer[i] = 0;
    }
}

template<typename T>
void tensor<T>::set_name(std::string name){
    this->name = name;
}

template<typename T>
T* tensor<T>::mem_alloc(type::backend_type backend, int64_t size){
    T* buffer = nullptr;

    if(backend == type::backend_type::CPU){
        buffer = (T *) malloc(size * sizeof(T));
    }
    return buffer;
}

template<typename T>
tensor<T>::~tensor(){
    free(this->buffer);
}

