// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2018 - 2019 Daniil Goncharov <neargye@gmail.com>.
//
// Permission is hereby  granted, free of charge, to any  person obtaining a copy
// of this software and associated  documentation files (the "Software"), to deal
// in the Software  without restriction, including without  limitation the rights
// to  use, copy,  modify, merge,  publish, distribute,  sublicense, and/or  sell
// copies  of  the Software,  and  to  permit persons  to  whom  the Software  is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE  IS PROVIDED "AS  IS", WITHOUT WARRANTY  OF ANY KIND,  EXPRESS OR
// IMPLIED,  INCLUDING BUT  NOT  LIMITED TO  THE  WARRANTIES OF  MERCHANTABILITY,
// FITNESS FOR  A PARTICULAR PURPOSE AND  NONINFRINGEMENT. IN NO EVENT  SHALL THE
// AUTHORS  OR COPYRIGHT  HOLDERS  BE  LIABLE FOR  ANY  CLAIM,  DAMAGES OR  OTHER
// LIABILITY, WHETHER IN AN ACTION OF  CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE  OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tf_utils.hpp"
#include "scope_guard.hpp"
#include <iostream>
#include <vector>

#include "TFUtil.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

int main() {

	TensorflowC TC;
	TC.outputnames.push_back("lambda_8/Reshape");
	TC.input_dims = { 1, 256, 256, 3 };

	TC.Load("saved_model.pb");

	for (size_t i = 1; i < 5; i++)
	{
		string imgfile = "img/" + to_string(i) + ".png";
		Mat orig = imread(imgfile);
		Mat img;
		orig.convertTo(img, CV_32F, 1.0 / 255);
		assert(img.cols == 256 && img.rows == 256 && img.channels() == 3);
		TC.SetData((float*)img.data);
		TC.Run();
		auto dims = TC.GetOutputDims();
		float* result = TC.GetData();

		assert(dims[0] == 1);
		for (size_t j = 0; j < dims[1] / 2; j++)
		{
			float x = result[j * 2 + 0] * (256 - 1);
			float y = result[j * 2 + 1] * (256 - 1);
			
			circle(orig, Point(x, y), 2, Scalar(0, 255, 0), -1);
		}

		imshow("IMG", orig);
		waitKey(1000);
		

	}
	
	TC.Close();

	auto graph = tf_utils::LoadGraph("saved_model.pb");
	SCOPE_EXIT{ tf_utils::DeleteGraph(graph); }; // Auto-delete on scope exit.
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		return 1;
	}

	auto status = TF_NewStatus();
	SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.

	auto names = GetOpNames(graph, status);


	auto input_op = TF_Output{ TF_GraphOperationByName(graph, names[0].c_str()), 0 };
	if (input_op.oper == nullptr) {
		std::cout << "Can't init input_op" << std::endl;
		return 2;
	}


	PrintTensorInfo(graph, names[0].c_str(), status);
	std::cout << std::endl;

	PrintTensorInfo(graph, names[names.size() - 1].c_str(), status);




	const std::vector<std::int64_t> input_dims = { 1, 207 };
	const std::vector<float> input_vals(207);


	auto input_tensor = tf_utils::CreateTensor(TF_FLOAT, input_dims, input_vals);
	SCOPE_EXIT{ tf_utils::DeleteTensor(input_tensor); }; // Auto-delete on scope exit.

	auto out_op = TF_Output{ TF_GraphOperationByName(graph, "lambda_8/Reshape"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't init out_op" << std::endl;
		return 3;
	}

	TF_Tensor* output_tensor = nullptr;

	//auto status = TF_NewStatus();
	//SCOPE_EXIT{ TF_DeleteStatus(status); }; // Auto-delete on scope exit.
	auto options = TF_NewSessionOptions();
	auto sess = TF_NewSession(graph, options, status);
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

	auto data = static_cast<float*>(TF_TensorData(output_tensor));

	std::cout << "Output vals: " << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << std::endl;
	getchar();
	return 0;
}