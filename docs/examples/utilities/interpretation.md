# Model Interpretation Examples

This page demonstrates model interpretation and visualization techniques for understanding predictions.

## Basic Interpretation

Visualize and interpret model predictions using various techniques.

```python
import autotimm as at  # recommended alias
from autotimm import ImageClassifier
from autotimm.interpretation import ModelInterpreter
import torch
from PIL import Image


def main():
    # Load trained model
    model = ImageClassifier.load_from_checkpoint("best-model.ckpt", compile_model=False)
    model.eval()
    
    # Create interpreter
    interpreter = ModelInterpreter(model)
    
    # Load and preprocess image
    image = Image.open("test_image.jpg")
    
    # Get prediction with interpretation
    result = interpreter.interpret(
        image,
        methods=["gradcam", "integrated_gradients", "occlusion"],
        target_layer="layer4",  # ResNet final layer
    )
    
    # Visualize results
    interpreter.visualize(
        image,
        result,
        save_path="interpretation_results.png",
        show_original=True,
    )
    
    # Print prediction details
    print(f"Predicted class: {result['class']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Attribution scores: {result['attributions']}")


if __name__ == "__main__":
    main()
```

**Interpretation Methods:**

| Method | Description | Best For |
|--------|-------------|----------|
| GradCAM | Gradient-based class activation mapping | Quick visualization |
| Integrated Gradients | Path integral of gradients | Precise attribution |
| Occlusion | Systematically mask input regions | Understanding importance |
| LIME | Local interpretable model approximation | Decision boundaries |
| SHAP | Shapley values for feature importance | Comprehensive analysis |

---

## Interpretation Metrics

Evaluate interpretation quality with quantitative metrics.

```python
from autotimm.interpretation import ModelInterpreter, InterpretationMetrics


def main():
    model = ImageClassifier.load_from_checkpoint("best-model.ckpt", compile_model=False)
    interpreter = ModelInterpreter(model)
    metrics = InterpretationMetrics()
    
    # Load test images
    images = ["img1.jpg", "img2.jpg", "img3.jpg"]
    
    results = []
    for img_path in images:
        image = Image.open(img_path)
        
        # Get interpretation
        result = interpreter.interpret(
            image,
            methods=["gradcam"],
            target_layer="layer4",
        )
        
        # Compute metrics
        scores = metrics.evaluate(
            result,
            metrics=["faithfulness", "sensitivity", "complexity"],
        )
        
        results.append({
            "image": img_path,
            "prediction": result["class"],
            "faithfulness": scores["faithfulness"],
            "sensitivity": scores["sensitivity"],
            "complexity": scores["complexity"],
        })
    
    # Summary statistics
    for r in results:
        print(f"{r['image']}: {r['prediction']}")
        print(f"  Faithfulness: {r['faithfulness']:.3f}")
        print(f"  Sensitivity: {r['sensitivity']:.3f}")
        print(f"  Complexity: {r['complexity']:.3f}")


if __name__ == "__main__":
    main()
```

**Interpretation Metrics:**

- **Faithfulness**: How well the interpretation reflects the model's actual reasoning
- **Sensitivity**: How much the interpretation changes with small input perturbations
- **Complexity**: Sparsity and simplicity of the interpretation
- **Stability**: Consistency of interpretations across similar inputs

---

## Interpretation Phase 2

Advanced interpretation techniques including counterfactual explanations.

```python
from autotimm.interpretation import ModelInterpreter, CounterfactualGenerator


def main():
    model = ImageClassifier.load_from_checkpoint("best-model.ckpt", compile_model=False)
    interpreter = ModelInterpreter(model)
    cf_generator = CounterfactualGenerator(model)
    
    image = Image.open("test_image.jpg")
    
    # Original prediction
    original_result = interpreter.interpret(image)
    print(f"Original: {original_result['class']} ({original_result['confidence']:.2%})")
    
    # Generate counterfactual
    counterfactual = cf_generator.generate(
        image,
        target_class="different_class",
        max_iterations=100,
        regularization=0.01,
    )
    
    # Compare original and counterfactual
    cf_result = interpreter.interpret(counterfactual["image"])
    print(f"Counterfactual: {cf_result['class']} ({cf_result['confidence']:.2%})")
    print(f"Distance: {counterfactual['distance']:.4f}")
    
    # Visualize changes
    interpreter.visualize_counterfactual(
        original_image=image,
        counterfactual_image=counterfactual["image"],
        save_path="counterfactual_comparison.png",
    )


if __name__ == "__main__":
    main()
```

**Counterfactual Explanations:**

- **What-If Analysis**: "What changes would flip the prediction?"
- **Minimal Perturbations**: Find smallest changes needed
- **Feature Importance**: Identify most influential features
- **Decision Boundaries**: Understand classification boundaries

---

## Interpretation Phase 3

Comprehensive interpretation with multiple techniques and analysis.

```python
from autotimm.interpretation import (
    ModelInterpreter,
    InterpretationMetrics,
    InterpretationComparison,
)


def main():
    model = ImageClassifier.load_from_checkpoint("best-model.ckpt", compile_model=False)
    interpreter = ModelInterpreter(model)
    metrics = InterpretationMetrics()
    comparison = InterpretationComparison()
    
    image = Image.open("test_image.jpg")
    
    # Run multiple interpretation methods
    methods = ["gradcam", "integrated_gradients", "occlusion", "lime", "shap"]
    results = {}
    
    for method in methods:
        result = interpreter.interpret(
            image,
            methods=[method],
            target_layer="layer4",
        )
        
        # Evaluate each method
        scores = metrics.evaluate(result)
        results[method] = {
            "interpretation": result,
            "scores": scores,
        }
    
    # Compare methods
    comparison_result = comparison.compare(
        results,
        metrics=["faithfulness", "sensitivity", "agreement"],
    )
    
    # Visualize comparison
    comparison.visualize_comparison(
        comparison_result,
        save_path="method_comparison.png",
    )
    
    # Print best method
    best_method = comparison_result["best_method"]
    print(f"Best interpretation method: {best_method}")
    print(f"Scores: {results[best_method]['scores']}")


if __name__ == "__main__":
    main()
```

**Method Comparison Criteria:**

- **Agreement**: How similar are different methods' results?
- **Consistency**: Do methods agree on important regions?
- **Computational Cost**: Time and memory requirements
- **Interpretability**: How easy to understand and explain?

---

## Interactive Visualization

Create interactive visualizations for exploring interpretations.

```python
from autotimm.interpretation import InteractiveVisualizer
import gradio as gr


def main():
    model = ImageClassifier.load_from_checkpoint("best-model.ckpt", compile_model=False)
    visualizer = InteractiveVisualizer(model)
    
    # Create interactive interface
    def interpret_image(image, method, target_layer):
        result = visualizer.interpret(
            image,
            method=method,
            target_layer=target_layer,
        )
        return visualizer.create_visualization(result)
    
    # Build Gradio interface
    interface = gr.Interface(
        fn=interpret_image,
        inputs=[
            gr.Image(type="pil", label="Input Image"),
            gr.Dropdown(
                choices=["gradcam", "integrated_gradients", "occlusion", "lime"],
                value="gradcam",
                label="Interpretation Method",
            ),
            gr.Dropdown(
                choices=["layer1", "layer2", "layer3", "layer4"],
                value="layer4",
                label="Target Layer",
            ),
        ],
        outputs=gr.Image(type="pil", label="Interpretation"),
        title="Model Interpretation Explorer",
        description="Explore different interpretation methods for image classification",
    )
    
    interface.launch(share=True)


if __name__ == "__main__":
    main()
```

**Interactive Features:**

- **Real-time interpretation**: See results as you change settings
- **Method comparison**: Switch between interpretation methods
- **Layer selection**: Explore different network layers
- **Parameter tuning**: Adjust interpretation parameters
- **Export results**: Save interpretations and visualizations

---

## Comprehensive Tutorial

The [`comprehensive_interpretation_tutorial.ipynb`](https://github.com/Noctillio-Ai/AutoTimm/blob/main/examples/interpretation/comprehensive_interpretation_tutorial.ipynb) notebook provides a complete guide to model interpretation.

**Topics Covered:**

1. **Setup and Installation**
   - Required dependencies
   - Model loading and preparation

2. **Basic Interpretation**
   - GradCAM visualization
   - Integrated Gradients
   - Occlusion sensitivity

3. **Advanced Techniques**
   - LIME explanations
   - SHAP values
   - Counterfactual generation

4. **Quantitative Evaluation**
   - Interpretation metrics
   - Method comparison
   - Statistical analysis

5. **Practical Applications**
   - Debugging models
   - Understanding failures
   - Building trust in predictions

6. **Interactive Tools**
   - Building dashboards
   - User interfaces
   - Real-time interpretation

**Running the Notebook:**

```bash
# Install Jupyter
pip install jupyter notebook

# Launch notebook
jupyter notebook examples/interpretation/comprehensive_interpretation_tutorial.ipynb
```

---

## Running Examples

```bash
python examples/interpretation/interpretation_demo.py
python examples/interpretation/interpretation_metrics_demo.py
python examples/interpretation/interpretation_phase2_demo.py
python examples/interpretation/interpretation_phase3_demo.py
python examples/interpretation/interactive_visualization_demo.py

# Run the comprehensive notebook
jupyter notebook examples/interpretation/comprehensive_interpretation_tutorial.ipynb
```

## See Also

- [Interpretation User Guide](../../user-guide/interpretation/index.md) - Full interpretation documentation
- [Interpretation Metrics Guide](../../user-guide/interpretation/metrics.md) - Interpretation metrics reference
- [Interactive Visualizations Guide](../../user-guide/interpretation/interactive-visualizations.md) - Visualization tools and techniques

