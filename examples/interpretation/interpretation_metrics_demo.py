"""
Explanation Quality Metrics Demo

Demonstrates how to quantitatively evaluate explanation methods using:
- Faithfulness metrics (deletion, insertion)
- Sensitivity analysis
- Sanity checks (parameter/data randomization)
- Localization metrics (pointing game)
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from autotimm import ImageClassifier
from autotimm.interpretation import (
    ExplanationMetrics,
    GradCAM,
    GradCAMPlusPlus,
    IntegratedGradients,
)


def create_sample_image():
    """Create a sample image for demonstration."""
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Create gradient pattern
    for i in range(224):
        img[i, :, 0] = int(255 * i / 224)
        img[:, i, 1] = int(255 * i / 224)
    img[:, :, 2] = 128
    return Image.fromarray(img)


def example_1_deletion_insertion():
    """Example 1: Faithfulness metrics (deletion/insertion)."""
    print("\n" + "=" * 60)
    print("Example 1: Faithfulness Metrics")
    print("=" * 60)

    # Create model and explainer
    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    metrics = ExplanationMetrics(model, explainer, use_cuda=False)

    # Create sample image
    image = create_sample_image()

    # Deletion metric
    print("\nComputing deletion metric...")
    deletion_result = metrics.deletion(image, steps=20)
    print(f"✓ Deletion AUC: {deletion_result['auc']:.3f}")
    print(f"  Final drop: {deletion_result['final_drop']:.2%}")
    print(f"  Original score: {deletion_result['original_score']:.4f}")

    # Insertion metric
    print("\nComputing insertion metric...")
    insertion_result = metrics.insertion(image, steps=20)
    print(f"✓ Insertion AUC: {insertion_result['auc']:.3f}")
    print(f"  Final rise: {insertion_result['final_rise']:.2%}")
    print(f"  Baseline score: {insertion_result['baseline_score']:.4f}")

    # Plot curves
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Deletion curve
    ax1.plot(deletion_result["scores"], "b-", linewidth=2)
    ax1.set_xlabel("Deletion Steps")
    ax1.set_ylabel("Prediction Score")
    ax1.set_title("Deletion Curve (Lower = Better)")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(
        y=deletion_result["original_score"],
        color="r",
        linestyle="--",
        label="Original",
        alpha=0.5,
    )
    ax1.legend()

    # Insertion curve
    ax2.plot(insertion_result["scores"], "g-", linewidth=2)
    ax2.set_xlabel("Insertion Steps")
    ax2.set_ylabel("Prediction Score")
    ax2.set_title("Insertion Curve (Higher = Better)")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(
        y=insertion_result["baseline_score"],
        color="r",
        linestyle="--",
        label="Baseline",
        alpha=0.5,
    )
    ax2.axhline(
        y=insertion_result["original_score"],
        color="b",
        linestyle="--",
        label="Original",
        alpha=0.5,
    )
    ax2.legend()

    plt.tight_layout()
    plt.savefig("metrics_faithfulness.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved faithfulness curves to: metrics_faithfulness.png")


def example_2_sensitivity():
    """Example 2: Sensitivity analysis."""
    print("\n" + "=" * 60)
    print("Example 2: Sensitivity Analysis")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    metrics = ExplanationMetrics(model, explainer, use_cuda=False)

    image = create_sample_image()

    # Compute sensitivity
    print("\nComputing sensitivity-n metric...")
    sensitivity_result = metrics.sensitivity_n(image, n_samples=30, noise_level=0.15)

    print(f"✓ Sensitivity: {sensitivity_result['sensitivity']:.4f}")
    print(f"  Std deviation: {sensitivity_result['std']:.4f}")
    print(f"  Max change: {sensitivity_result['max_change']:.4f}")

    # Plot distribution of changes
    plt.figure(figsize=(8, 5))
    plt.hist(sensitivity_result["changes"], bins=15, edgecolor="black", alpha=0.7)
    plt.xlabel("Explanation Change")
    plt.ylabel("Frequency")
    plt.title(
        f'Sensitivity Distribution\nMean: {sensitivity_result["sensitivity"]:.4f}'
    )
    plt.axvline(
        x=sensitivity_result["sensitivity"],
        color="r",
        linestyle="--",
        label="Mean",
        linewidth=2,
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("metrics_sensitivity.png", dpi=150, bbox_inches="tight")
    print("\n✓ Saved sensitivity distribution to: metrics_sensitivity.png")

    # Interpretation
    if sensitivity_result["sensitivity"] < 0.05:
        print("\n✓ Interpretation: Very stable (low sensitivity)")
    elif sensitivity_result["sensitivity"] < 0.15:
        print("\n✓ Interpretation: Moderately stable")
    else:
        print("\n⚠ Interpretation: High sensitivity (unstable explanations)")


def example_3_sanity_checks():
    """Example 3: Sanity checks."""
    print("\n" + "=" * 60)
    print("Example 3: Sanity Checks")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    metrics = ExplanationMetrics(model, explainer, use_cuda=False)

    image = create_sample_image()

    # Model parameter randomization test
    print("\nRunning model parameter randomization test...")
    param_result = metrics.model_parameter_randomization_test(image)

    print(f"✓ Correlation with randomized model: {param_result['correlation']:.3f}")
    print(f"  Mean change: {param_result['change']:.3f}")
    print(f"  Test passed: {param_result['passes']}")

    if param_result["passes"]:
        print("  → Explanation is sensitive to model parameters ✓")
    else:
        print("  → WARNING: Explanation not sensitive to model parameters!")

    # Data randomization test
    print("\nRunning data randomization test...")
    data_result = metrics.data_randomization_test(image)

    if not np.isnan(data_result["correlation"]):
        print(f"✓ Correlation with random class: {data_result['correlation']:.3f}")
    else:
        print("✓ Correlation with random class: N/A (constant heatmaps)")
    print(f"  Mean change: {data_result['change']:.3f}")
    print(f"  Test passed: {data_result['passes']}")

    if data_result["passes"]:
        print("  → Explanation is sensitive to target class ✓")
    else:
        print("  → WARNING: Explanation not sensitive to target class!")


def example_4_pointing_game():
    """Example 4: Pointing game (localization)."""
    print("\n" + "=" * 60)
    print("Example 4: Pointing Game (Localization)")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    metrics = ExplanationMetrics(model, explainer, use_cuda=False)

    image = create_sample_image()

    # Define ground truth bounding box
    # (In real use, this would be from annotations)
    bbox = [75, 75, 150, 150]

    print("\nRunning pointing game...")
    print(f"Ground truth bbox: {bbox}")

    result = metrics.pointing_game(image, bbox)

    print(f"✓ Max attention location: {result['max_location']}")
    print(f"  Hit (inside bbox): {result['hit']}")

    if result["hit"]:
        print("  → Explanation correctly localizes object ✓")
    else:
        print("  → Explanation misses object location")


def example_5_compare_methods():
    """Example 5: Compare multiple explanation methods."""
    print("\n" + "=" * 60)
    print("Example 5: Compare Explanation Methods")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    image = create_sample_image()

    # Test multiple methods
    methods = {
        "GradCAM": GradCAM(model),
        "GradCAM++": GradCAMPlusPlus(model),
        "IntegratedGradients": IntegratedGradients(model, baseline="black", steps=30),
    }

    print("\nComparing methods (deletion & insertion)...")
    print(f"{'Method':<20} {'Del AUC':<10} {'Ins AUC':<10} {'Sensitivity':<12}")
    print("-" * 60)

    for method_name, explainer in methods.items():
        metrics_obj = ExplanationMetrics(model, explainer, use_cuda=False)

        # Compute key metrics
        deletion = metrics_obj.deletion(image, steps=10)
        insertion = metrics_obj.insertion(image, steps=10)
        sensitivity = metrics_obj.sensitivity_n(image, n_samples=10)

        print(
            f"{method_name:<20} {deletion['auc']:<10.3f} {insertion['auc']:<10.3f} "
            f"{sensitivity['sensitivity']:<12.4f}"
        )

    print("\nInterpretation:")
    print("  - Lower deletion AUC = better (drops faster)")
    print("  - Higher insertion AUC = better (rises faster)")
    print("  - Lower sensitivity = more stable")


def example_6_evaluate_all():
    """Example 6: Run all metrics at once."""
    print("\n" + "=" * 60)
    print("Example 6: Comprehensive Evaluation")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    metrics = ExplanationMetrics(model, explainer, use_cuda=False)

    image = create_sample_image()
    bbox = [75, 75, 150, 150]

    print("\nRunning comprehensive evaluation...")
    print("This may take a minute...\n")

    results = metrics.evaluate_all(image, bbox=bbox)

    # Print summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print("\n1. Faithfulness:")
    print(f"   Deletion AUC: {results['deletion']['auc']:.3f}")
    print(f"   Insertion AUC: {results['insertion']['auc']:.3f}")

    print("\n2. Sensitivity:")
    print(f"   Sensitivity-n: {results['sensitivity']['sensitivity']:.4f}")

    print("\n3. Sanity Checks:")
    print(
        f"   Param randomization: {'✓ PASS' if results['param_randomization']['passes'] else '✗ FAIL'}"
    )
    print(
        f"   Data randomization: {'✓ PASS' if results['data_randomization']['passes'] else '✗ FAIL'}"
    )

    print("\n4. Localization:")
    print(
        f"   Pointing game: {'✓ HIT' if results['pointing_game']['hit'] else '✗ MISS'}"
    )

    # Overall assessment
    print("\n" + "=" * 60)
    score = 0
    if results["deletion"]["auc"] < 0.8:
        score += 1
    if results["insertion"]["auc"] > 0.6:
        score += 1
    if results["sensitivity"]["sensitivity"] < 0.15:
        score += 1
    if results["param_randomization"]["passes"]:
        score += 1
    if results["data_randomization"]["passes"]:
        score += 1
    if results["pointing_game"]["hit"]:
        score += 1

    print(f"Overall Score: {score}/6")
    if score >= 5:
        print("✓ Excellent explanation quality!")
    elif score >= 3:
        print("✓ Good explanation quality")
    else:
        print("⚠ Consider improving explanation method")


def main():
    """Run all metrics examples."""
    print("\n" + "=" * 60)
    print("Explanation Quality Metrics Demo")
    print("=" * 60)

    # Run examples
    example_1_deletion_insertion()
    example_2_sensitivity()
    example_3_sanity_checks()
    example_4_pointing_game()
    example_5_compare_methods()
    example_6_evaluate_all()

    print("\n" + "=" * 60)
    print("✓ All examples completed!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - metrics_faithfulness.png")
    print("  - metrics_sensitivity.png")
    print("\nMetrics computed:")
    print("  ✓ Deletion (faithfulness)")
    print("  ✓ Insertion (faithfulness)")
    print("  ✓ Sensitivity-n (stability)")
    print("  ✓ Model parameter randomization (sanity check)")
    print("  ✓ Data randomization (sanity check)")
    print("  ✓ Pointing game (localization)")
    print("\nUse these metrics to:")
    print("  1. Evaluate explanation method quality")
    print("  2. Compare different explanation methods")
    print("  3. Validate that explanations are meaningful")
    print("  4. Debug explanation methods")


if __name__ == "__main__":
    main()
