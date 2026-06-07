#include <onnx_test.hpp>

TEST_CASE(meanvariancenormalization_default_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<std::size_t> input_lens{1, 3, 4, 5};
    auto input_type = migraphx::shape::float_type;
    migraphx::shape s{input_type, input_lens};
    auto x = mm->add_parameter("x", s);
    
    std::vector<int64_t> axes = {0, 2, 3};
    // auto sum = mm->add_instruction(migraphx::make_op("reduce_sum"), x);
    // float numElements = 60.0f;

    // auto denom = mm->add_literal(
    //         migraphx::literal{
    //             migraphx::shape{input_type},
    //             {numElements}
    //         });
    // auto mean = mm->add_instruction(migraphx::make_op("div"), sum, denom);
    // auto xCentered = mm->add_instruction(migraphx::make_op("sub"), x, mean);

    // auto xCenteredSquared = mm->add_instruction(migraphx::make_op("mul"), xCentered, xCentered);
    // auto xCenSum = mm->add_instruction(migraphx::make_op("reduce_sum"), xCenteredSquared);
    // auto xCenterSquaredMean = mm->add_instruction(migraphx::make_op("div"), xCenSum, denom);
    // auto sqrt = mm->add_instruction(migraphx::make_op("sqrt"), xCenterSquaredMean);

    // mm->add_instruction(migraphx::make_op("div"), xCentered, sqrt);

    auto mean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
    auto mb_mean = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), mean);
    auto xCentered = mm->add_instruction(migraphx::make_op("sub"), x, mb_mean);
    
    auto xCenteredSquared = mm->add_instruction(migraphx::make_op("mul"), xCentered, xCentered);
    auto xCenterSquaredMean = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), xCenteredSquared);
    auto mb_xCenSqMean = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), xCenterSquaredMean);
    auto std_dev = mm->add_instruction(migraphx::make_op("sqrt"), mb_xCenSqMean);

    mm->add_instruction(migraphx::make_op("div"), xCentered, std_dev);

    auto prog = optimize_onnx("mean_variance_normalization_default_test.onnx");
    EXPECT(p == prog);
}
