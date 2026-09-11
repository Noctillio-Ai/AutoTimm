"""Example: Deploy TorchScript model in C++ environment.

This example shows how to export a model and provides C++ code for deployment.
The exported model can be loaded and used in C++ applications without Python.
"""

import torch
from autotimm import ImageClassifier, export_to_torchscript


def export_for_cpp():
    """Export model for C++ deployment."""
    print("=" * 70)
    print("Exporting Model for C++ Deployment")
    print("=" * 70)

    # Create and export model
    model = ImageClassifier(
        backbone="resnet50",
        num_classes=1000,  # ImageNet classes
    )
    model.eval()

    # Export with inference optimization
    example_input = torch.randn(1, 3, 224, 224)
    export_to_torchscript(
        model=model,
        save_path="model_for_cpp.pt",
        example_input=example_input,
        optimize=True,
    )

    print("✓ Model exported: model_for_cpp.pt")
    print("\n" + "=" * 70)
    print("C++ Deployment Code")
    print("=" * 70)

    cpp_code = """
// File: inference.cpp
// Compile: g++ -std=c++17 inference.cpp -o inference \\
//          `pkg-config --cflags --libs opencv4` \\
//          -I${TORCH_PATH}/include \\
//          -L${TORCH_PATH}/lib -ltorch -ltorch_cpu -lc10

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <memory>
#include <vector>

class ImageClassifier {
public:
    ImageClassifier(const std::string& model_path) {
        try {
            // Load the TorchScript model
            module_ = torch::jit::load(model_path);
            module_.eval();
            std::cout << "Model loaded successfully\\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << "\\n";
            throw;
        }
    }

    torch::Tensor preprocess(const cv::Mat& image) {
        // Resize to 224x224
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(224, 224));

        // Convert BGR to RGB
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

        // Convert to float and normalize
        resized.convertTo(resized, CV_32FC3, 1.0 / 255.0);

        // Normalize with ImageNet stats
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};

        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < resized.rows; i++) {
                for (int j = 0; j < resized.cols; j++) {
                    resized.at<cv::Vec3f>(i, j)[c] =
                        (resized.at<cv::Vec3f>(i, j)[c] - mean[c]) / std[c];
                }
            }
        }

        // Convert to tensor [1, 3, 224, 224]
        auto tensor = torch::from_blob(
            resized.data,
            {1, resized.rows, resized.cols, 3},
            torch::kFloat32
        );
        tensor = tensor.permute({0, 3, 1, 2}).contiguous();

        return tensor;
    }

    std::pair<int, float> predict(const cv::Mat& image) {
        // Preprocess
        auto input_tensor = preprocess(image);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);

        torch::NoGradGuard no_grad;
        auto output = module_.forward(inputs).toTensor();

        // Get prediction
        auto probabilities = torch::softmax(output, 1);
        auto max_result = probabilities.max(1);
        auto max_prob = std::get<0>(max_result).item<float>();
        auto max_index = std::get<1>(max_result).item<int>();

        return {max_index, max_prob};
    }

private:
    torch::jit::script::Module module_;
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <model.pt> <image.jpg>\\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    try {
        // Load model
        ImageClassifier classifier(model_path);

        // Load image
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            std::cerr << "Error: Could not load image\\n";
            return 1;
        }

        // Run prediction
        auto [class_id, confidence] = classifier.predict(image);

        std::cout << "Predicted class: " << class_id << "\\n";
        std::cout << "Confidence: " << confidence * 100 << "%\\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\\n";
        return 1;
    }
}
"""

    print(cpp_code)

    print("\n" + "=" * 70)
    print("Build Instructions")
    print("=" * 70)
    print("""
1. Install LibTorch:
   wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip
   unzip libtorch-shared-with-deps-latest.zip

2. Install OpenCV:
   # Ubuntu/Debian
   sudo apt-get install libopencv-dev

   # macOS
   brew install opencv

3. Set environment variable:
   export TORCH_PATH=/path/to/libtorch

4. Compile:
   g++ -std=c++17 inference.cpp -o inference \\
       `pkg-config --cflags --libs opencv4` \\
       -I${TORCH_PATH}/include \\
       -L${TORCH_PATH}/lib -ltorch -ltorch_cpu -lc10

5. Run:
   ./inference model_for_cpp.pt image.jpg
    """)

    print("=" * 70)
    print("CMakeLists.txt Example")
    print("=" * 70)

    cmake_code = """
cmake_minimum_required(VERSION 3.18)
project(autotimm_inference)

set(CMAKE_CXX_STANDARD 17)

# Find Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find OpenCV
find_package(OpenCV REQUIRED)

# Add executable
add_executable(inference inference.cpp)

# Link libraries
target_link_libraries(inference ${TORCH_LIBRARIES} ${OpenCV_LIBS})
"""

    print(cmake_code)

    print("\nBuild with CMake:")
    print("  mkdir build && cd build")
    print("  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..")
    print("  cmake --build . --config Release")


def export_for_mobile():
    """Export model for mobile deployment."""
    print("\n" + "=" * 70)
    print("Exporting Model for Mobile Deployment")
    print("=" * 70)

    model = ImageClassifier(
        backbone="mobilenet_v3_small",  # Lightweight model for mobile
        num_classes=1000,
    )
    model.eval()

    example_input = torch.randn(1, 3, 224, 224)
    export_to_torchscript(
        model=model,
        save_path="mobile_model.pt",
        example_input=example_input,
        optimize=True,
    )

    print("✓ Mobile model exported: mobile_model.pt")
    print("\nAndroid Integration (Java):")
    print("""
// Add to build.gradle
dependencies {
    implementation 'org.pytorch:pytorch_android:1.13.1'
    implementation 'org.pytorch:pytorch_android_torchvision:1.13.1'
}

// Java code
import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

public class ImageClassifier {
    private Module module;

    public ImageClassifier(String modelPath) {
        module = Module.load(modelPath);
    }

    public int predict(Bitmap bitmap) {
        // Preprocess bitmap to tensor
        Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            new float[]{0.485f, 0.456f, 0.406f},  // mean
            new float[]{0.229f, 0.224f, 0.225f}   // std
        );

        // Run inference
        Tensor outputTensor = module.forward(IValue.from(inputTensor))
                                    .toTensor();

        // Get prediction
        float[] scores = outputTensor.getDataAsFloatArray();
        int maxIndex = 0;
        float maxScore = scores[0];
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }
}
    """)

    print("\niOS Integration (Swift):")
    print("""
import TorchModule

class ImageClassifier {
    private var module: TorchModule

    init(modelPath: String) throws {
        module = try TorchModule(fileAtPath: modelPath)
    }

    func predict(image: UIImage) -> Int {
        // Preprocess image
        guard let pixelBuffer = image.pixelBuffer(
            width: 224,
            height: 224
        ) else {
            return -1
        }

        // Convert to tensor
        guard let tensor = TorchTensor.from(pixelBuffer: pixelBuffer) else {
            return -1
        }

        // Run inference
        guard let output = module.forward(tensor) else {
            return -1
        }

        // Get prediction
        let scores = output.floatArray
        let maxIndex = scores.enumerated()
            .max(by: { $0.element < $1.element })?.offset ?? 0

        return maxIndex
    }
}
    """)


def main():
    """Run deployment examples."""
    export_for_cpp()
    export_for_mobile()

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
✓ Exported models can be deployed to:
  - C++ applications (desktop, server)
  - Mobile devices (iOS, Android)
  - Edge devices (Raspberry Pi, Jetson Nano)
  - Embedded systems

✓ No Python dependency required in production
✓ Fast inference with LibTorch
✓ Single-file deployment

Next steps:
1. Export your trained model: model.to_torchscript('model.pt')
2. Choose your deployment platform
3. Integrate using the code examples above
4. Deploy to production!

For more examples:
- docs/user-guide/deployment/cpp-deployment.md
- docs/user-guide/deployment/mobile-deployment.md
    """)


if __name__ == "__main__":
    main()
