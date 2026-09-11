"""
Performance Optimization Demo

Demonstrates performance optimization techniques for model interpretations:
- Caching for faster repeated explanations
- Batch processing for multiple images
- Performance profiling to identify bottlenecks
- Model optimization for inference
"""

import time

import numpy as np
import torch
from PIL import Image

from autotimm import ImageClassifier
from autotimm.interpretation import GradCAM
from autotimm.interpretation.optimization import (
    BatchProcessor,
    ExplanationCache,
    PerformanceProfiler,
    optimize_for_inference,
)


def create_sample_image(seed=None):
    """Create a sample image for demonstration."""
    if seed is not None:
        np.random.seed(seed)

    img = np.random.rand(224, 224, 3)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def example_1_caching():
    """Example 1: Caching for faster repeated explanations."""
    print("\n" + "=" * 60)
    print("Example 1: Explanation Caching")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    image = create_sample_image(seed=42)

    # Without caching
    print("\nWithout caching:")
    start = time.time()
    for i in range(5):
        heatmap = explainer.explain(image, target_class=5)
    time_without_cache = time.time() - start
    print(
        f"✓ 5 explanations: {time_without_cache:.3f}s ({time_without_cache/5:.3f}s each)"
    )

    # With caching
    print("\nWith caching:")
    cache = ExplanationCache(cache_dir="./demo_cache", max_size_mb=100)

    start = time.time()
    for i in range(5):
        # Check cache first
        heatmap = cache.get(image, method="gradcam", target_class=5)
        if heatmap is None:
            # Compute and cache
            heatmap = explainer.explain(image, target_class=5)
            cache.put(image, method="gradcam", explanation=heatmap, target_class=5)
    time_with_cache = time.time() - start
    print(f"✓ 5 explanations: {time_with_cache:.3f}s ({time_with_cache/5:.3f}s each)")

    speedup = time_without_cache / time_with_cache
    print(f"\n✓ Speedup: {speedup:.1f}x faster with caching!")

    # Cache statistics
    stats = cache.stats()
    print("\nCache statistics:")
    print(f"  Entries: {stats['num_entries']}")
    print(f"  Size: {stats['total_size_mb']:.2f} MB")
    print(f"  Utilization: {stats['utilization']:.1%}")

    # Cleanup
    cache.clear()


def example_2_batch_processing():
    """Example 2: Batch processing for multiple images."""
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)

    # Create test images
    n_images = 20
    images = [create_sample_image(seed=i) for i in range(n_images)]

    # Sequential processing (one-by-one)
    print(f"\nProcessing {n_images} images sequentially:")
    start = time.time()
    heatmaps_seq = []
    for img in images:
        heatmap = explainer.explain(img)
        heatmaps_seq.append(heatmap)
    time_sequential = time.time() - start
    print(f"✓ Time: {time_sequential:.3f}s ({time_sequential/n_images:.3f}s per image)")

    # Batch processing
    print(f"\nProcessing {n_images} images with batch processor:")
    processor = BatchProcessor(
        model, explainer, batch_size=8, show_progress=False, use_cuda=False
    )

    start = time.time()
    heatmaps_batch = processor.process_batch(images)
    time_batch = time.time() - start
    print(f"✓ Time: {time_batch:.3f}s ({time_batch/n_images:.3f}s per image)")

    speedup = time_sequential / time_batch
    print(f"\n✓ Speedup: {speedup:.1f}x faster with batching!")

    # Verify results are similar
    diff = np.mean(
        [np.abs(h1 - h2).mean() for h1, h2 in zip(heatmaps_seq, heatmaps_batch)]
    )
    print(f"\nResult difference: {diff:.6f} (should be ~0)")


def example_3_profiling():
    """Example 3: Performance profiling."""
    print("\n" + "=" * 60)
    print("Example 3: Performance Profiling")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    profiler = PerformanceProfiler(enabled=True)

    image = create_sample_image(seed=42)

    print("\nProfiling interpretation pipeline...")

    # Profile different stages
    with profiler.profile("total"):
        with profiler.profile("preprocessing"):
            # Simulate preprocessing
            tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0)

        with profiler.profile("forward_pass"), torch.inference_mode():
            # Forward pass
            model(tensor)

        with profiler.profile("explanation"):
            # Generate explanation
            heatmap = explainer.explain(image, target_class=5)

        with profiler.profile("postprocessing"):
            # Simulate postprocessing
            (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    # Print statistics
    profiler.print_stats()

    # Identify bottleneck
    stats = profiler.get_stats()
    slowest = max(stats.items(), key=lambda x: x[1]["mean"])
    print(f"\n⚠ Bottleneck: {slowest[0]} ({slowest[1]['mean']:.3f}s)")


def example_4_model_optimization():
    """Example 4: Model optimization for inference."""
    print("\n" + "=" * 60)
    print("Example 4: Model Optimization")
    print("=" * 60)

    image = create_sample_image(seed=42)
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)

    # Original model
    print("\nOriginal model:")
    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    start = time.time()
    for _ in range(10):
        with torch.inference_mode():
            model(tensor)
    time_original = time.time() - start
    print(f"✓ 10 forward passes: {time_original:.3f}s ({time_original/10:.3f}s each)")

    # Optimized model
    print("\nOptimized model:")
    model_opt = optimize_for_inference(model, use_fp16=False)

    start = time.time()
    for _ in range(10):
        with torch.inference_mode():
            model_opt(tensor)
    time_optimized = time.time() - start
    print(f"✓ 10 forward passes: {time_optimized:.3f}s ({time_optimized/10:.3f}s each)")

    speedup = time_original / time_optimized
    print(f"\n✓ Speedup: {speedup:.1f}x faster with optimization!")

    print("\nOptimizations applied:")
    print("  ✓ Disabled gradient computation")
    print("  ✓ Enabled cudnn benchmarking")
    print("  ✓ Set model to eval mode")


def example_5_cache_management():
    """Example 5: Cache management and statistics."""
    print("\n" + "=" * 60)
    print("Example 5: Cache Management")
    print("=" * 60)

    model = ImageClassifier(backbone="resnet18", num_classes=10)
    model.eval()

    explainer = GradCAM(model)
    cache = ExplanationCache(cache_dir="./demo_cache", max_size_mb=10)  # Small cache

    print("\nPopulating cache...")
    images = [create_sample_image(seed=i) for i in range(50)]

    for i, img in enumerate(images[:20]):
        heatmap = explainer.explain(img, target_class=5)
        cache.put(img, method="gradcam", explanation=heatmap, target_class=5)

        if (i + 1) % 5 == 0:
            stats = cache.stats()
            print(
                f"After {i+1} images: {stats['num_entries']} entries, "
                f"{stats['total_size_mb']:.2f} MB ({stats['utilization']:.1%} full)"
            )

    print("\n✓ Cache automatically evicts old entries when full (LRU policy)")

    # Test cache hits
    print("\nTesting cache retrieval:")
    hits = 0
    misses = 0

    for img in images[:10]:  # Test first 10 (some may be evicted)
        heatmap = cache.get(img, method="gradcam", target_class=5)
        if heatmap is not None:
            hits += 1
        else:
            misses += 1

    print(f"✓ Cache hits: {hits}")
    print(f"✓ Cache misses: {misses}")
    print(f"  Hit rate: {hits/(hits+misses):.1%}")

    # Cleanup
    cache.clear()
    print("\n✓ Cache cleared")


def example_6_production_tips():
    """Example 6: Production deployment tips."""
    print("\n" + "=" * 60)
    print("Example 6: Production Deployment Tips")
    print("=" * 60)

    print("\nRecommended optimization strategy:")
    print()
    print("1. Model Optimization:")
    print("   ```python")
    print("   model = optimize_for_inference(model, use_fp16=True)  # GPU only")
    print("   ```")
    print()
    print("2. Enable Caching:")
    print("   ```python")
    print(
        "   cache = ExplanationCache(cache_dir='/var/cache/explanations', max_size_mb=5000)"
    )
    print("   ```")
    print()
    print("3. Use Batch Processing:")
    print("   ```python")
    print("   processor = BatchProcessor(model, explainer, batch_size=32)")
    print("   heatmaps = processor.process_batch(images)")
    print("   ```")
    print()
    print("4. Profile in Production:")
    print("   ```python")
    print("   profiler = PerformanceProfiler(enabled=True)")
    print("   with profiler.profile('request'):")
    print("       explanation = generate_explanation(image)")
    print("   if profiler.get_stats()['request']['mean'] > 1.0:")
    print("       log_warning('Slow explanation detected')")
    print("   ```")
    print()
    print("Expected performance improvements:")
    print("  • Caching: 10-50x faster for repeated images")
    print("  • Batching: 2-5x faster for multiple images")
    print("  • Model optimization: 1.5-3x faster inference")
    print("  • Combined: Up to 100x faster overall!")


def main():
    """Run all performance optimization examples."""
    print("\n" + "=" * 60)
    print("Performance Optimization Demo")
    print("=" * 60)

    try:
        example_1_caching()
        example_2_batch_processing()
        example_3_profiling()
        example_4_model_optimization()
        example_5_cache_management()
        example_6_production_tips()

        print("\n" + "=" * 60)
        print("✓ All examples completed!")
        print("=" * 60)
        print("\nPerformance optimizations demonstrated:")
        print("  ✓ Explanation caching (10-50x speedup)")
        print("  ✓ Batch processing (2-5x speedup)")
        print("  ✓ Performance profiling")
        print("  ✓ Model optimization (1.5-3x speedup)")
        print("  ✓ Cache management")
        print("\nKey takeaways:")
        print("  • Use caching for repeated explanations")
        print("  • Process multiple images in batches")
        print("  • Profile to identify bottlenecks")
        print("  • Optimize model for inference")
        print("  • Monitor cache hit rates")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
