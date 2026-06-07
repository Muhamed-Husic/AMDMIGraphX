#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {
struct parse_mean_variance_normalization : op_parser<parse_mean_variance_normalization>
{
    std::vector<op_desc> operators() const { return {{"MeanVarianceNormalization"}}; }
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        // 1. Extract the 'axes' attribute or default to [0, 2, 3]
        std::vector<int64_t> axes = {0, 2, 3};
        if(contains(info.attributes, "axes"))
        {
            literal s = parser.parse_value(info.attributes.at("axes"));
            s.visit([&](auto v) { axes.assign(v.begin(), v.end()); });
        }
        auto x = args[0];
        auto x_type = x->get_shape().type();
        auto x_lens = x->get_shape().lens();
        // 2. Compute the mean
        auto mean = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
        auto mean_bcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), mean);
        // 3. Subtract the mean: (X - E[X])
        auto sub = info.add_instruction(migraphx::make_op("sub"), x, mean_bcast);
        // 4. Square the difference, reduce to find variance, and broadcast
        auto sq = info.add_instruction(migraphx::make_op("mul"), sub, sub);
        auto var = info.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sq);
        auto var_bcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), var);
        // 5. Create the epsilon literal safely considering float16 data types
        float epsilon = 1e-9f;
        epsilon = (x_type == migraphx::shape::half_type && std::abs(epsilon) < 1e-7f) ? 1e-7f : epsilon;
        auto eps_lit = info.add_literal(migraphx::literal{migraphx::shape{x_type, {1}}, {epsilon}});
        auto eps_bcast = info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), eps_lit);
        
        // 6. Final math: add epsilon, sqrt, and divide
        auto var_eps = info.add_instruction(migraphx::make_op("add"), var_bcast, eps_bcast);
        auto std_dev = info.add_instruction(migraphx::make_op("sqrt"), var_eps);
        return info.add_instruction(migraphx::make_op("div"), sub, std_dev);
    }
};
} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx