#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(mean_variance_normalization_verify_test)
{
    migraphx::program p = read_onnx("mean_variance_normalization_axes_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {1, 3, 4, 5}};
    std::vector<float> data = {
        // Channel 0
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,

        // Channel 1
        10,11,12,13,14,
        10,11,12,13,14,
        10,11,12,13,14,
        10,11,12,13,14,

        // Channel 2
        20,21,22,23,24,
        20,21,22,23,24,
        20,21,22,23,24,
        20,21,22,23,24
    };

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = data;
    constexpr int BATCH = 1;
    constexpr int CCHANNEL = 3;
    constexpr int WIDTH = 4;
    constexpr int HEIGHT = 5;
    constexpr int NO_ELEMENTS = 20; // HEIGHT * WIDTH
    int count{};
    // this is only for batch = 1
    for(int c=0; c<CCHANNEL; ++c) {
        float sum{};
        
        // sum
        for(size_t i=count; i < count + NO_ELEMENTS; ++i) {
            sum += data[i];
        }
        float mean = sum / NO_ELEMENTS;
        
        // variance
        sum = 0.0f;
        for(size_t i=count; i<count + NO_ELEMENTS; ++i) {
            gold[i] = data[i] - mean;
            gold[i] *= gold[i];
            sum += gold[i];
        }
        float variance = sum / NO_ELEMENTS;
        float std_dev = std::sqrt(variance);

        // normalization
        for(size_t i = count; i < count +  NO_ELEMENTS; ++i) {
            gold[i] = (data[i] - mean) / std_dev;
        }

        count += NO_ELEMENTS;
    }
    
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}