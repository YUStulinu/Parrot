// Helpers of the C API: errors become return codes and a thread-local message.
#pragma once

#include <exception>
#include <string>
#include <vector>

#include "phones.hpp"

namespace parrot::api {

extern thread_local std::string last_error;

template <class F>
int guard(F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        last_error = e.what();
        return -1;
    }
}

template <class F>
char* guard_str(F&& f) {
    try {
        return f();
    } catch (const std::exception& e) {
        last_error = e.what();
        return nullptr;
    }
}

const PhoneMap* vocab();
void set_vocab(PhoneMap m);
std::vector<int> parse_phones(const char* phones);
char* dup(const std::string& s);

}  // namespace parrot::api
