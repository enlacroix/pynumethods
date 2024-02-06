#include <iostream>
#include <limits>

int main() {
    setlocale(LC_ALL, "Russian");
    int k = 0;
    float a = 1;
    double b = 1;
    long double c = 1;

    while (a != 0) {
        a /= 2;
        k++;
    }
    std::cout << "[float] Нуль = 2^-" << k << std::endl;
    k = 0;
    a = 1;
    while (a < std::numeric_limits<float>::max()) {
        a *= 2;
        k++;
    }
    std::cout << "[float] Бесконечность = 2^" << k << std::endl;
    k = 0;
    a = 1;
    while (1 + a > 1) {
        a /= 2;
        k++;
    }
    std::cout << "[float] Эпсилон = 2^-" << k << std::endl;
    std::cout << std::endl;
    k = 0;
    while (b != 0) {
        b /= 2;
        k++;
    }
    std::cout << "[double] Нуль = 2^-" << k << std::endl;
    k = 0;
    b = 1;
    while (b < std::numeric_limits<double>::max()) {
        b *= 2;
        k++;
    }
    std::cout << "[double] Бесконечность = 2^" << k << std::endl;
    k = 0;
    b = 1;
    while (1 + b > 1) {
        b /= 2;
        k++;
    }
    std::cout << "[double] Эпсилон = 2^-" << k << std::endl;
    std::cout << std::endl;
    k = 0;
    while (c != 0) {
        c /= 2;
        k++;
    }
    std::cout << std::endl;
    std::cout << "[long double] Нуль = 2^-" << k << std::endl;
    k = 0;
    c = 1;
    while (c < std::numeric_limits<long double>::max()) {
        c *= 2;
        k++;
    }
    std::cout << "[long double] Бесконечность = 2^" << k << std::endl;
    k = 0;
    c = 1;
    while (1 + c > 1) {
        c /= 2;
        k++;
    }
    std::cout << "[long double] Эпсилон = 2^-" << k << std::endl;
    return 0;
}