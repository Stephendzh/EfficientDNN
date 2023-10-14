//
// Created by HP on 2023/10/14.
//

#include <iostream>
#include <chrono>

using namespace std;


const int height_feature = 56;
const int width_feature = 56;
const int input_channels = 3;
const int output_channels = 64;
const int kernel_size = 3;
const int step_width = width_feature - kernel_size + 1;
const int step_height = height_feature - kernel_size + 1;
const int num_element = step_height * step_width * input_channels * kernel_size * kernel_size;
const int max_num = 20;
const int min_num = 10;
// 加入数组元素的大小限制防止在卷积过程中整数值的大小溢出
int input_feature_map[input_channels][width_feature][kernel_size];
int kernel[input_channels][kernel_size][kernel_size];
int flat_kernel[input_channels * kernel_size * kernel_size];
int conv_element[num_element];
int output_feature_map[step_height][step_width];
int img2col_ifm[step_height * step_width][input_channels * kernel_size * kernel_size];

//初始化 input_feature_map
void init_ifm() {
    for (int i = 0; i < height_feature; ++i) {
        for (int j = 0; j < width_feature; ++j) {
            for (int k = 0; k < input_channels; ++k) {
                input_feature_map[i][j][k] = rand() % (max_num - min_num + 1) + min_num;
            }
        }
    }
}

//初始化kernel的一个filter，注意，本次作业中有64个filter
void init_kernel() {
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int k = 0; k < input_channels; ++k) {
                kernel[i][j][k] = rand() % (max_num - min_num + 1) + min_num;
            }
        }
    }
}

//把input_feature_map中的每个元素按照img2col的逻辑取出
void element() {
    int count = 0;
    for (int i = 0; i < step_height; ++i) {
        for (int j = 0; j < step_width; ++j) {
            for (int c = 0; c < input_channels; ++c) {
                for (int k1 = 0; k1 < kernel_size; ++k1) {
                    for (int k2 = 0; k2 < kernel_size; ++k2) {
                        conv_element[count] = input_feature_map[c][i + k1][j + k2];
                        ++count;
                    }
                }
            }
        }
    }
}

// 适用于img2col卷积，展平3 * 3 *3的kernel，获得展平的kernel的一个filter，运行64次的时间0毫秒，可以忽略不计
void get_flat_kernel() {
    int count = 0;
    for (int i = 0; i < input_channels; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            for (int k = 0; k < kernel_size; ++k) {
                flat_kernel[count] = kernel[i][j][k];
                ++count;
            }
        }
    }
}

//把取出的元素放入img2col的input_feature_map中，这里每一行是一个卷积核卷积的元素
void get_img2col_ifm() {
    int count = 0;
    for (int j = 0; j < (step_width * step_height); ++j) {
        for (int i = 0; i < (input_channels * kernel_size * kernel_size); ++i) {
            ++count;
            img2col_ifm[j][i] = conv_element[count - 1];
        }
    }
}

void img2col_convolution() {
    int count = 0;
    for (int i = 0; i < step_height; ++i) {
        for (int j = 0; j < step_width; ++j) {
            for (int k = 0; k < input_channels * kernel_size * kernel_size; ++k) {
                output_feature_map[i][j] += img2col_ifm[count][k] * flat_kernel[k];
            }
            ++count;
        }
    }
}
void img2col_convolution_unrolling() {
    int count = 0;
    for (int i = 0; i < step_height; ++i) {
        for (int j = 0; j < step_width; ++j) {
            for (int k = 0; k < input_channels * kernel_size * kernel_size / 3; ++k) {
                output_feature_map[i][j] += img2col_ifm[count][3*k] * flat_kernel[3*k];
                output_feature_map[i][j] += img2col_ifm[count][3*k+1] * flat_kernel[3*k+1];
                output_feature_map[i][j] += img2col_ifm[count][3*k+2] * flat_kernel[3*k+2];
            }
            ++count;
        }
    }
}

void show_ofm() {
    for (int i = 0; i < step_height; ++i) {
        for (int j = 0; j < step_width; ++j) {
            cout << output_feature_map[i][j] << "  ";
        }
    }
}

void whole_img2col() {
    for (int i = 0; i < output_channels; ++i) {
        init_kernel();
        get_flat_kernel();
        img2col_convolution();
    }
}

void whole_img2col_unrolling() {
    for (int i = 0; i < output_channels; ++i) {
        init_kernel();
        get_flat_kernel();
        img2col_convolution_unrolling();
    }
}

int main() {
    init_ifm(); //初始化input_feature_map
    //init_kernel(); //初始化卷积核的一个filter
    element(); //以img2col的逻辑读取input_feature_map元素
    get_img2col_ifm(); //获得二维的img2col input_feature_map
    //get_flat_kernel(); //获得展平的kernel的一个filter
    auto start_time = chrono::high_resolution_clock::now();     // 获取当前时间点，作为程序开始时间
    // 在这里执行您的程序
    whole_img2col();
    //show_ofm();
    auto end_time = chrono::high_resolution_clock::now();    // 获取当前时间点，作为程序结束时间

    auto duration = chrono::duration_cast<chrono::microseconds>(end_time - start_time);    // 计算程序运行时间（微秒为单位）
    // cout << kernel << endl;
    cout << "running time:" << duration.count() << "microseconds" << endl;

    auto start_time1 = chrono::high_resolution_clock::now();     // 获取当前时间点，作为程序开始时间
    // 在这里执行您的程序
    whole_img2col_unrolling();
    //show_ofm();
    auto end_time1 = chrono::high_resolution_clock::now();    // 获取当前时间点，作为程序结束时间

    auto duration1 = chrono::duration_cast<chrono::microseconds>(end_time - start_time);    // 计算程序运行时间（微秒为单位）
    // cout << kernel << endl;
    cout << "running time:" << duration1.count() << "microseconds" << endl;
    return 0;
}
