#pragma once

#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace std;
inline void PrintOpInputs(TF_Graph*, TF_Operation* op) {
	auto num_inputs = TF_OperationNumInputs(op);

	std::cout << "Number inputs: " << num_inputs << std::endl;

	for (auto i = 0; i < num_inputs; ++i) {
		auto input = TF_Input{ op, i };
		auto type = TF_OperationInputType(input);
		std::cout << std::to_string(i) << " type : " << tf_utils::DataTypeToString(type) << std::endl;
	}
}

inline void PrintOpOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status) {
	auto num_outputs = TF_OperationNumOutputs(op);

	std::cout << "Number outputs: " << num_outputs << std::endl;

	for (auto i = 0; i < num_outputs; ++i) {
		auto output = TF_Output{ op, i };
		auto type = TF_OperationOutputType(output);
		std::cout << std::to_string(i) << " type : " << tf_utils::DataTypeToString(type);

		auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Can't get tensor dimensionality" << std::endl;
			continue;
		}

		std::cout << " dims: " << num_dims;

		if (num_dims <= 0) {
			std::cout << " []" << std::endl;;
			continue;
		}

		std::vector<std::int64_t> dims(num_dims);
		TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Can't get get tensor shape" << std::endl;
			continue;
		}

		std::cout << " [";
		for (auto j = 0; j < num_dims; ++j) {
			std::cout << dims[j];
			if (j < num_dims - 1) {
				std::cout << ",";
			}
		}
		std::cout << "]" << std::endl;
	}
}


inline void PrintInputs(TF_Graph*, TF_Operation* op) {
	auto num_inputs = TF_OperationNumInputs(op);

	for (auto i = 0; i < num_inputs; ++i) {
		auto input = TF_Input{ op, i };
		auto type = TF_OperationInputType(input);
		std::cout << "Input: " << i << " type: " << tf_utils::DataTypeToString(type) << std::endl;
	}
}

inline void PrintOutputs(TF_Graph* graph, TF_Operation* op, TF_Status* status) {
	auto num_outputs = TF_OperationNumOutputs(op);

	for (int i = 0; i < num_outputs; ++i) {
		auto output = TF_Output{ op, i };
		auto type = TF_OperationOutputType(output);
		auto num_dims = TF_GraphGetTensorNumDims(graph, output, status);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Can't get tensor dimensionality" << std::endl;
			continue;
		}

		std::cout << " dims: " << num_dims;

		if (num_dims <= 0) {
			std::cout << " []" << std::endl;;
			continue;
		}

		std::vector<std::int64_t> dims(num_dims);

		std::cout << "Output: " << i << " type: " << tf_utils::DataTypeToString(type);
		TF_GraphGetTensorShape(graph, output, dims.data(), num_dims, status);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Can't get get tensor shape" << std::endl;
			continue;
		}

		std::cout << " [";
		for (auto d = 0; d < num_dims; ++d) {
			std::cout << dims[d];
			if (d < num_dims - 1) {
				std::cout << ", ";
			}
		}
		std::cout << "]" << std::endl;
	}
}

inline void PrintTensorInfo(TF_Graph* graph, const char* layer_name, TF_Status* status) {
	std::cout << "Tensor: " << layer_name;
	auto op = TF_GraphOperationByName(graph, layer_name);

	if (op == nullptr) {
		std::cout << "Could not get " << layer_name << std::endl;
		return;
	}

	auto num_inputs = TF_OperationNumInputs(op);
	auto num_outputs = TF_OperationNumOutputs(op);
	std::cout << " inputs: " << num_inputs << " outputs: " << num_outputs << std::endl;

	PrintInputs(graph, op);

	PrintOutputs(graph, op, status);
}


inline std::vector<std::string> GetOpNames(TF_Graph* graph, TF_Status* status, bool print=false) {
	TF_Operation* op;
	std::size_t pos = 0;

	std::vector<std::string> output;

	while ((op = TF_GraphNextOperation(graph, &pos)) != nullptr) {
		auto name = TF_OperationName(op);
		auto type = TF_OperationOpType(op);
		auto device = TF_OperationDevice(op);

		auto num_outputs = TF_OperationNumOutputs(op);
		auto num_inputs = TF_OperationNumInputs(op);

		if(print)std::cout << pos << ": " << name << " type: " << type << " device: " << device << " number inputs: " << num_inputs << " number outputs: " << num_outputs << std::endl;

		output.push_back(name);
	}

	return output;
}


class TensorflowC
{
public:
	TF_Session* sess;
	TF_Graph* graph;
	TF_Status* status;
	TF_Output input_op;
	TF_Output out_op;

	TF_Tensor* input_tensor = nullptr;
	TF_Tensor* output_tensor = nullptr;

	vector<string> inputnames;
	vector<string> outputnames;

	std::vector<std::int64_t> input_dims;
	std::vector<float> input_vals;


	int Load(string modelfile)
	{
		graph = tf_utils::LoadGraph(modelfile.c_str());
		if (graph == nullptr) {
			std::cout << "Can't load graph" << std::endl;
			return 1;
		}

		auto names = GetOpNames(graph, status);

		if (inputnames.size() == 0)
		{
			inputnames.push_back(names[0]);
		}

		//TODO Support multi input.
		input_op = TF_Output{ TF_GraphOperationByName(graph, inputnames[0].c_str()), 0 };
		if (input_op.oper == nullptr) {
			std::cout << "Can't init input_op" << std::endl;
			return 2;
		}

		std::int64_t size = 1;
		for (size_t i = 0; i < input_dims.size(); i++)
		{
			size *= input_dims[i];
		}
		input_vals = std::vector<float>(size);


		input_tensor = tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals);

		/*PrintTensorInfo(graph, names[0].c_str(), status);
		std::cout << std::endl;

		PrintTensorInfo(graph, names[names.size() - 1].c_str(), status);*/
		if (outputnames.size() == 0)
		{
			std::cout << "No output tensor" << std::endl;
			return 9999;
		}

		out_op = TF_Output{ TF_GraphOperationByName(graph, outputnames[0].c_str()), 0 };
		if (out_op.oper == nullptr) {
			std::cout << "Can't init out_op" << std::endl;
			return 3;
		}



		status = TF_NewStatus();
		//SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
		auto options = TF_NewSessionOptions();
		sess = TF_NewSession(graph, options, status);
		TF_DeleteSessionOptions(options);

		if (TF_GetCode(status) != TF_OK) {
			return 4;
		}


		TF_SessionRun(sess,
			nullptr, // Run options.
			&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
			&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
			nullptr, 0, // Target operations, number of targets.
			nullptr, // Run metadata.
			status // Output status.
		);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Error run session";
			return 5;
		}

	}

	std::vector<std::int64_t> GetOutputDims()
	{
		int N = TF_NumDims(output_tensor);
		auto output = std::vector<std::int64_t>(N);

		for (size_t i = 0; i < N; i++)
		{
			output[i] = TF_Dim(output_tensor, i);
			cout << output[i] << ",";
		}
		cout << endl;
		return output;
	}

	void SetData(float *ptr)
	{
		if(input_tensor != nullptr)
			tf_utils::DeleteTensor(input_tensor);

		memcpy(input_vals.data(), ptr, input_vals.size() * sizeof(float));
		input_tensor = tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals);
	}

	float* GetData()
	{
		return static_cast<float*>(TF_TensorData(output_tensor));
	}

	int Run()
	{
		//status = TF_NewStatus();
		TF_SessionRun(sess,
			nullptr, // Run options.
			&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
			&out_op, &output_tensor, 1, // Output tensors, output tensor values, number of outputs.
			nullptr, 0, // Target operations, number of targets.
			nullptr, // Run metadata.
			status // Output status.
		);

		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Error run session";
			return 5;
		}

		/*auto data = static_cast<float*>(TF_TensorData(output_tensor));

		std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;*/
	}

	int Close()
	{
		tf_utils::DeleteGraph(graph);
		tf_utils::DeleteTensor(input_tensor);

		TF_CloseSession(sess, status);
		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Error close session";
			return 6;
		}

		TF_DeleteSession(sess, status);
		if (TF_GetCode(status) != TF_OK) {
			std::cout << "Error delete session";
			return 7;
		}

		TF_DeleteStatus(status);
	}
};

