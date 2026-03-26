import torch
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple, List
from collections import defaultdict
import warnings

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


class CheckpointInspector:
    """Comprehensive checkpoint structure inspector with detailed analysis"""

    def __init__(
        self,
        checkpoint_path: str,
        weights_only: bool = False,
        verbose: bool = True
    ):
        self.checkpoint_path = checkpoint_path
        self.weights_only = weights_only
        self.verbose = verbose
        self.checkpoint = None
        self.state_dict = None
        self.analysis_cache = {}

    def run(self) -> bool:
        """Run full inspection pipeline"""
        print("=" * 80)
        print("CHECKPOINT INSPECTOR")
        print("=" * 80)

        if not self._validate_file():
            return False

        if not self._load_checkpoint():
            return False

        if not self._extract_state_dict():
            return False

        self._print_parameters_by_group()
        self._print_detailed_dtype_analysis()
        self._print_shape_patterns()
        self._print_tensor_statistics()
        self._print_architecture_analysis()
        self._print_model_complexity()
        self._print_warnings_and_notes()
        self._print_summary()

        print("=" * 80)
        return True

    def _validate_file(self) -> bool:
        """Validate checkpoint file exists and is readable"""
        print(f"\nChecking: {self.checkpoint_path}\n")

        if not os.path.exists(self.checkpoint_path):
            print(f"❌ File not found: {self.checkpoint_path}")
            return False

        try:
            size_mb = os.path.getsize(self.checkpoint_path) / (1024 * 1024)
            print(f"✅ File found (size: {size_mb:.2f} MB)")
            return True
        except Exception as e:
            print(f"❌ Cannot access file: {e}")
            return False

    def _load_checkpoint(self) -> bool:
        """Load checkpoint from disk"""
        print(f"Loading checkpoint...\n")

        try:
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=self.weights_only,
            )
            print("✅ Checkpoint loaded successfully\n")
            return True
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {type(e).__name__}: {e}")
            return False

    def _extract_state_dict(self) -> bool:
        """Extract state_dict from checkpoint"""
        if isinstance(self.checkpoint, dict):
            # Try common keys
            for key in ["params", "model_state_dict", "state_dict"]:
                if key in self.checkpoint:
                    self.state_dict = self.checkpoint[key]
                    print(f"State dict source: '{key}' key")
                    break
            else:
                self.state_dict = self.checkpoint
                print(f"State dict source: root level")

            print(f"Checkpoint keys: {list(self.checkpoint.keys())}\n")
        else:
            self.state_dict = self.checkpoint
            print(f"Checkpoint type: {type(self.checkpoint).__name__}\n")

        if not self.state_dict:
            print("❌ State dict is empty")
            return False

        if not isinstance(self.state_dict, dict):
            print(f"❌ State dict is not a dict: {type(self.state_dict)}")
            return False

        print(f"State dict size: {len(self.state_dict)} parameters\n")
        return True

    def _print_parameters_by_group(self):
        """Group and print parameters by prefix"""
        print("=" * 80)
        print("PARAMETERS BY GROUP:")
        print("=" * 80)

        groups = self._group_by_prefix()

        for prefix in sorted(groups.keys()):
            keys = groups[prefix]
            total_params = self._count_params_in_keys(keys)
            print(f"\n{prefix} ({len(keys)} tensors, {total_params:,} params)")
            print("-" * 80)

            # Show first 15 entries
            for key in sorted(keys)[:15]:
                self._print_tensor_info(key)

            if len(keys) > 15:
                print(f"  ... and {len(keys) - 15} more")

    def _print_detailed_dtype_analysis(self):
        """Detailed dtype analysis with memory usage"""
        print("\n" + "=" * 80)
        print("DATA TYPE ANALYSIS:")
        print("=" * 80)

        dtype_stats = defaultdict(
            lambda: {"count": 0, "memory_bytes": 0, "param_count": 0}
        )

        for tensor in tqdm(
            self.state_dict.values(),
            desc="Analyzing dtypes",
            disable=not self.verbose
        ):
            if not self._is_valid_tensor(tensor):
                continue

            dtype_str = str(tensor.dtype)
            numel = tensor.numel()
            elem_size = tensor.element_size()

            dtype_stats[dtype_str]["count"] += 1
            dtype_stats[dtype_str]["memory_bytes"] += numel * elem_size
            dtype_stats[dtype_str]["param_count"] += numel

        print("\nDtype breakdown:")
        print("-" * 80)

        total_memory = sum(s["memory_bytes"] for s in dtype_stats.values())

        for dtype in sorted(dtype_stats.keys()):
            stats = dtype_stats[dtype]
            memory_mb = stats["memory_bytes"] / (1024 ** 2)
            percentage = (
                (stats["memory_bytes"] / total_memory * 100)
                if total_memory > 0
                else 0
            )

            print(
                f"  {dtype:15s} | "
                f"{stats['count']:5d} tensors | "
                f"{stats['param_count']:12,d} params | "
                f"{memory_mb:8.2f} MB ({percentage:5.1f}%)"
            )

        print(
            f"\nTotal memory: {total_memory / (1024**2):.2f} MB "
            f"({total_memory / (1024**3):.3f} GB)"
        )

    def _print_shape_patterns(self):
        """Analyze and print tensor shape patterns"""
        print("\n" + "=" * 80)
        print("SHAPE PATTERNS:")
        print("=" * 80)

        shape_patterns = defaultdict(list)

        for key, tensor in tqdm(
            self.state_dict.items(),
            desc="Analyzing shapes",
            disable=not self.verbose
        ):
            if not self._is_valid_tensor(tensor):
                continue

            shape = tuple(tensor.shape)
            shape_patterns[shape].append(key)

        # Sort by frequency
        sorted_patterns = sorted(
            shape_patterns.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        print(f"\nTop 20 shape patterns:")
        print("-" * 80)

        for i, (shape, keys) in enumerate(sorted_patterns[:20]):
            sample_key = keys[0]
            sample_tensor = self.state_dict[sample_key]
            dtype = str(sample_tensor.dtype).split(".")[-1]
            numel = sample_tensor.numel()

            print(
                f"  {str(shape):30s} | "
                f"{len(keys):4d} tensors | "
                f"{dtype:8s} | "
                f"{numel:,d} params"
            )

        if len(sorted_patterns) > 20:
            print(
                f"\n  ... and {len(sorted_patterns) - 20} more unique shapes"
            )

    def _print_tensor_statistics(self):
        """Print statistical analysis of tensor values"""
        print("\n" + "=" * 80)
        print("TENSOR VALUE STATISTICS:")
        print("=" * 80)

        stats = {
            "sparse": [],
            "zero_heavy": [],
            "extreme_values": [],
        }

        for key, tensor in tqdm(
            self.state_dict.items(),
            desc="Analyzing values",
            disable=not self.verbose
        ):
            if not self._is_valid_tensor(tensor):
                continue

            try:
                # Check for sparse tensors
                if isinstance(tensor, torch.sparse.FloatTensor):
                    stats["sparse"].append(key)
                    continue

                tensor_float = tensor.float()
                numel = tensor.numel()

                # Check for zero-heavy tensors (>50% zeros)
                zero_count = (tensor_float == 0).sum().item()
                zero_ratio = zero_count / numel if numel > 0 else 0

                if zero_ratio > 0.5:
                    stats["zero_heavy"].append(
                        (key, f"{zero_ratio*100:.1f}%")
                    )

                # Check for extreme values
                if numel > 0:
                    min_val = tensor_float.min().item()
                    max_val = tensor_float.max().item()

                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        stats["extreme_values"].append(
                            (key, f"[{min_val:.2e}, {max_val:.2e}]")
                        )

            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error analyzing {key}: {e}")

        # Print findings
        if stats["sparse"]:
            print(f"\nSparse tensors ({len(stats['sparse'])}):")
            for key in stats["sparse"][:5]:
                print(f"  - {key}")
            if len(stats["sparse"]) > 5:
                print(f"  ... and {len(stats['sparse']) - 5} more")

        if stats["zero_heavy"]:
            print(f"\nZero-heavy tensors (>50% zeros) ({len(stats['zero_heavy'])}):")
            for key, ratio in stats["zero_heavy"][:5]:
                print(f"  - {key}: {ratio}")
            if len(stats["zero_heavy"]) > 5:
                print(f"  ... and {len(stats['zero_heavy']) - 5} more")

        if stats["extreme_values"]:
            print(f"\nExtreme values ({len(stats['extreme_values'])}):")
            for key, range_str in stats["extreme_values"][:5]:
                print(f"  - {key}: {range_str}")
            if len(stats["extreme_values"]) > 5:
                print(f"  ... and {len(stats['extreme_values']) - 5} more")

    def _print_architecture_analysis(self):
        """Analyze and print architecture structure"""
        print("\n" + "=" * 80)
        print("ARCHITECTURE ANALYSIS:")
        print("=" * 80)

        # Analyze main body layers
        self._analyze_body_layers()

        # Analyze output layers
        self._analyze_output_layers()

        # Detect special layer types
        self._detect_special_layers()

        # Suggest architecture
        self._suggest_architecture()

    def _analyze_body_layers(self):
        """Analyze body layer structure"""
        body_keys = [k for k in self.state_dict.keys()
                     if k.startswith("body.")]

        if not body_keys:
            print("\n⚠️  No 'body.*' layers found")
            return

        print(f"\nBody layers: {len(body_keys)} tensors")

        # Find all numeric indices
        body_indices = self._extract_layer_indices(body_keys, "body")

        if not body_indices:
            print("⚠️  No numeric body layer indices found")
            return

        max_idx = max(body_indices)
        print(f"Layer indices: 0 to {max_idx} ({len(body_indices)} total)")

        print("\nBody structure (first 25 layers):")
        print("-" * 80)

        for i in range(min(max_idx + 1, 25)):
            layer_keys = [k for k in body_keys if k.startswith(f"body.{i}.")]

            if not layer_keys:
                continue

            # Get layer attributes
            attrs = set()
            for k in layer_keys:
                parts = k.split(".")
                if len(parts) > 2:
                    attrs.add(parts[2])

            attrs_str = ", ".join(sorted(attrs))
            print(f"  body.{i}: {attrs_str}")

            # Show shape of first weight
            weight_keys = [k for k in layer_keys if k.endswith("weight")]
            if weight_keys:
                first_key = weight_keys[0]
                try:
                    tensor = self.state_dict[first_key]
                    if self._is_valid_tensor(tensor):
                        shape = tensor.shape
                        print(f"           └─ {first_key} → {tuple(shape)}")
                except Exception as e:
                    print(
                        f"           └─ {first_key} "
                        f"(error: {type(e).__name__})"
                    )

        if max_idx >= 25:
            print(f"  ... and {max_idx - 24} more layers")

    def _analyze_output_layers(self):
        """Analyze upsampling and output layers"""
        output_prefixes = [
            "upconv", "upsample", "up", "hr", "conv_last",
            "output", "final", "tail", "reconstruction"
        ]

        print("\n" + "-" * 80)
        print("Output/Upsampling layers:")
        print("-" * 80)

        found_any = False
        for prefix in output_prefixes:
            matching = [
                k for k in self.state_dict.keys()
                if k.split(".")[0].lower() == prefix.lower()
            ]

            if matching:
                found_any = True
                print(f"\n{prefix}:")
                for key in sorted(matching)[:10]:
                    self._print_tensor_info(key)
                if len(matching) > 10:
                    print(f"  ... and {len(matching) - 10} more")

        if not found_any:
            print("\n⚠️  No output/upsampling layers found")

    def _detect_special_layers(self):
        """Detect and report special layer types"""
        print("\n" + "-" * 80)
        print("Special layer detection:")
        print("-" * 80)

        special_types = {
            "attention": [],
            "normalization": [],
            "embedding": [],
            "residual": [],
        }

        for key in self.state_dict.keys():
            key_lower = key.lower()

            # Attention
            if any(x in key_lower for x in ["attn", "attention", "query",
                                             "key", "value"]):
                special_types["attention"].append(key)

            # Normalization
            elif any(x in key_lower for x in ["norm", "bn", "ln", "gn"]):
                special_types["normalization"].append(key)

            # Embedding
            elif any(x in key_lower for x in ["embed", "token"]):
                special_types["embedding"].append(key)

            # Residual/skip connections (inferred from structure)
            elif "residual" in key_lower or "skip" in key_lower:
                special_types["residual"].append(key)

        for layer_type, keys in special_types.items():
            if keys:
                print(f"\n{layer_type.title()}: {len(keys)} tensors")
                for key in keys[:5]:
                    self._print_tensor_info(key)
                if len(keys) > 5:
                    print(f"  ... and {len(keys) - 5} more")

    def _suggest_architecture(self):
        """Suggest model architecture based on structure"""
        print("\n" + "=" * 80)
        print("ARCHITECTURE SUGGESTION:")
        print("=" * 80)

        # Check first conv
        if "body.0.weight" in self.state_dict:
            self._analyze_first_layer()

        # Analyze body pattern
        self._analyze_body_pattern()

        # Detect layer types
        self._detect_layer_count_types()

        # Estimate network depth
        self._estimate_network_depth()

    def _analyze_first_layer(self):
        """Analyze first convolutional layer"""
        try:
            weight = self.state_dict["body.0.weight"]
            if not self._is_valid_tensor(weight):
                return

            shape = weight.shape
            if len(shape) == 4:
                out_c, in_c, kh, kw = shape
                print(f"\nFirst layer (body.0):")
                print(f"  Conv2d({in_c}, {out_c}, kernel_size=({kh}, {kw}))")

                if in_c == 3:
                    print(f"  ✅ Takes RGB input (in_channels=3)")
                elif in_c == 1:
                    print(f"  ⚠️  Takes grayscale input (in_channels=1)")
                else:
                    print(f"  ℹ️  in_channels={in_c} (custom input)")
        except Exception as e:
            print(f"\n⚠️  Cannot analyze first layer: {e}")

    def _analyze_body_pattern(self):
        """Analyze repeating body pattern"""
        body_keys = [k for k in self.state_dict.keys()
                     if k.startswith("body.")]

        if not body_keys:
            return

        body_indices = self._extract_layer_indices(body_keys, "body")

        if len(body_indices) < 2:
            return

        # Check if pattern repeats
        sample_i = min(body_indices)
        pattern = self._get_layer_attributes(
            f"body.{sample_i}", body_keys
        )

        print(f"\nBody layer pattern (body.{sample_i}):")
        if pattern:
            print(f"  {', '.join(sorted(pattern))}")

    def _detect_layer_count_types(self):
        """Detect and count different layer types"""
        print(f"\nLayer type analysis:")

        body_keys = [k for k in self.state_dict.keys()
                     if k.startswith("body.")]

        if not body_keys:
            print("  ⚠️  No body layers to analyze")
            return

        body_indices = self._extract_layer_indices(body_keys, "body")

        conv_layers = 0
        norm_layers = 0
        attention_layers = 0

        for i in body_indices:
            layer_keys = [k for k in body_keys if k.startswith(f"body.{i}.")]

            has_weight = any(k.endswith("weight") for k in layer_keys)
            has_norm = any(
                any(x in k.lower() for x in ["norm", "bn", "ln", "gn"])
                for k in layer_keys
            )
            has_attention = any(
                any(x in k.lower() for x in ["attn", "attention"])
                for k in layer_keys
            )

            if has_attention:
                attention_layers += 1
            elif has_norm:
                norm_layers += 1
            elif has_weight:
                conv_layers += 1

        print(f"  Conv layers: {conv_layers}")
        print(f"  Norm layers: {norm_layers}")
        print(f"  Attention layers: {attention_layers}")

    def _estimate_network_depth(self):
        """Estimate network depth and characteristics"""
        body_indices = self._extract_layer_indices(
            [k for k in self.state_dict.keys() if k.startswith("body.")],
            "body"
        )

        if body_indices:
            depth = len(body_indices)
            print(f"\nEstimated depth: {depth} layers")

            if depth < 10:
                complexity = "shallow"
            elif depth < 50:
                complexity = "moderate"
            elif depth < 100:
                complexity = "deep"
            else:
                complexity = "very deep"

            print(f"Complexity estimate: {complexity}")

    def _print_model_complexity(self):
        """Estimate model complexity metrics"""
        print("\n" + "=" * 80)
        print("MODEL COMPLEXITY ESTIMATION:")
        print("=" * 80)

        # Analyze Conv2D layers for receptive field estimate
        print("\nConvolutional layer analysis:")
        print("-" * 80)

        conv_layers = []
        for key, tensor in self.state_dict.items():
            if not key.endswith("weight"):
                continue
            if not self._is_valid_tensor(tensor):
                continue

            shape = tensor.shape
            if len(shape) == 4:  # Conv2D: (out_c, in_c, kh, kw)
                out_c, in_c, kh, kw = shape
                conv_layers.append({
                    "name": key,
                    "out_c": out_c,
                    "in_c": in_c,
                    "kh": kh,
                    "kw": kw,
                })

        if conv_layers:
            print(f"Total Conv2D layers: {len(conv_layers)}")

            # Show some statistics
            kernel_sizes = [l["kh"] for l in conv_layers]
            out_channels = [l["out_c"] for l in conv_layers]

            if kernel_sizes:
                avg_kernel = sum(kernel_sizes) / len(kernel_sizes)
                max_kernel = max(kernel_sizes)
                print(f"  Kernel sizes: avg={avg_kernel:.1f}, max={max_kernel}")

            if out_channels:
                avg_channels = sum(out_channels) / len(out_channels)
                max_channels = max(out_channels)
                print(f"  Channels: avg={avg_channels:.0f}, max={max_channels}")

            # Estimate receptive field (simplified)
            receptive_field = 1
            stride = 1
            for layer in conv_layers:
                receptive_field += (layer["kh"] - 1) * stride
                # This is simplified; doesn't account for actual strides

            print(f"  Est. receptive field: ~{receptive_field}px")

    def _print_warnings_and_notes(self):
        """Print warnings and compatibility notes"""
        print("\n" + "=" * 80)
        print("WARNINGS & COMPATIBILITY NOTES:")
        print("=" * 80)

        warnings_list = []

        # Check for naming inconsistencies
        prefixes = set()
        for key in self.state_dict.keys():
            prefix = key.split(".")[0]
            prefixes.add(prefix)

        # Warn about unexpected prefixes
        common_prefixes = {
            "body", "head", "backbone", "encoder", "decoder",
            "upconv", "output", "conv_first", "conv_last"
        }

        unexpected = prefixes - common_prefixes
        if unexpected:
            warnings_list.append(
                f"Unexpected top-level modules: {', '.join(sorted(unexpected))}"
            )

        # Check for empty or malformed tensors
        for key, tensor in self.state_dict.items():
            if not self._is_valid_tensor(tensor):
                warnings_list.append(f"Invalid tensor: {key}")
                break  # Just report first one

        # Check for mixed precision
        dtypes = set()
        for tensor in self.state_dict.values():
            if hasattr(tensor, "dtype"):
                dtypes.add(str(tensor.dtype))

        if len(dtypes) > 2:
            warnings_list.append(
                f"Mixed precision detected: {', '.join(sorted(dtypes))}"
            )

        # Check input channel compatibility
        if "body.0.weight" in self.state_dict:
            try:
                first_layer = self.state_dict["body.0.weight"]
                if self._is_valid_tensor(first_layer):
                    in_c = first_layer.shape[1]
                    if in_c not in [1, 3, 4]:
                        warnings_list.append(
                            f"Unusual input channels: {in_c} "
                            f"(typically 1, 3, or 4)"
                        )
            except Exception:
                pass

        if warnings_list:
            for i, warning in enumerate(warnings_list, 1):
                print(f"\n{i}. ⚠️  {warning}")
        else:
            print("\n✅ No major compatibility issues detected")

    def _print_summary(self):
        """Print checkpoint summary statistics"""
        print("\n" + "=" * 80)
        print("SUMMARY:")
        print("=" * 80)

        # Total parameters
        try:
            total_params = self._count_all_params()
            print(f"\nTotal parameters: {total_params:,}")

            # Estimate for common model sizes
            if total_params < 1e6:
                size_category = "Very small (<1M)"
            elif total_params < 10e6:
                size_category = "Small (1-10M)"
            elif total_params < 50e6:
                size_category = "Medium (10-50M)"
            elif total_params < 100e6:
                size_category = "Large (50-100M)"
            else:
                size_category = "Very large (>100M)"

            print(f"Model size category: {size_category}")

        except Exception as e:
            print(f"\n⚠️  Cannot count parameters: {e}")

        # Key statistics
        print(f"\nKey statistics:")
        print(f"  Total tensors: {len(self.state_dict)}")

        groups = self._group_by_prefix()
        print(f"  Top-level groups: {len(groups)}")

        # Memory estimate
        try:
            total_bytes = 0
            for tensor in self.state_dict.values():
                if self._is_valid_tensor(tensor):
                    total_bytes += (
                        tensor.numel() * tensor.element_size()
                    )

            total_mb = total_bytes / (1024 ** 2)
            total_gb = total_bytes / (1024 ** 3)

            print(f"  Memory footprint: {total_mb:.2f} MB ({total_gb:.3f} GB)")
        except Exception as e:
            print(f"  Memory estimate: Unable to calculate ({e})")

        # Group sizes
        print(f"\nGroup sizes:")
        for prefix in sorted(groups.keys()):
            keys = groups[prefix]
            params = self._count_params_in_keys(keys)
            print(f"  {prefix}: {len(keys)} tensors, {params:,} params")

    def _group_by_prefix(self) -> Dict[str, list]:
        """Group state dict keys by first prefix"""
        groups = {}
        for key in self.state_dict.keys():
            prefix = key.split(".")[0]
            if prefix not in groups:
                groups[prefix] = []
            groups[prefix].append(key)
        return groups

    def _print_tensor_info(self, key: str):
        """Print tensor shape, dtype, and parameter count"""
        try:
            tensor = self.state_dict[key]

            if not self._is_valid_tensor(tensor):
                print(f"  {key:50s} (invalid tensor)")
                return

            dtype = str(tensor.dtype).split(".")[-1]
            numel = tensor.numel()
            numel_str = f"{numel:,}" if isinstance(numel, int) else str(numel)
            shape_str = str(tuple(tensor.shape))

            print(
                f"  {key:50s} {shape_str:25s} "
                f"{dtype:10s} ({numel_str})"
            )
        except Exception as e:
            print(
                f"  {key:50s} "
                f"(error: {type(e).__name__}: {str(e)[:30]})"
            )

    def _count_params_in_keys(self, keys: List[str]) -> int:
        """Count total parameters in a list of keys"""
        total = 0
        for key in keys:
            try:
                tensor = self.state_dict[key]
                if self._is_valid_tensor(tensor):
                    total += tensor.numel()
            except Exception:
                pass
        return total

    def _count_all_params(self) -> int:
        """Count all parameters in state dict"""
        total = 0
        for tensor in self.state_dict.values():
            if self._is_valid_tensor(tensor):
                total += tensor.numel()
        return total

    @staticmethod
    def _is_valid_tensor(obj: Any) -> bool:
        """Check if object is a valid tensor with numel"""
        try:
            return (
                isinstance(obj, (torch.Tensor, torch.nn.Parameter))
                or (hasattr(obj, "numel") and hasattr(obj, "shape")
                    and hasattr(obj, "dtype"))
            )
        except Exception:
            return False

    @staticmethod
    def _extract_layer_indices(keys: List[str], prefix: str) -> Set[int]:
        """Extract numeric layer indices from keys"""
        indices = set()
        prefix_with_dot = f"{prefix}."

        for key in keys:
            if not key.startswith(prefix_with_dot):
                continue

            parts = key.split(".")
            if len(parts) >= 2 and parts[1].isdigit():
                indices.add(int(parts[1]))

        return indices

    def _get_layer_attributes(
        self,
        layer_prefix: str,
        keys: List[str]
    ) -> Set[str]:
        """Get all attributes of a specific layer"""
        attrs = set()
        prefix_with_dot = f"{layer_prefix}."

        for key in keys:
            if key.startswith(prefix_with_dot):
                parts = key.split(".")
                if len(parts) > 2:
                    attrs.add(parts[2])

        return attrs


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoint structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inspect_checkpoint.py --checkpoint model.pth
  python inspect_checkpoint.py --checkpoint model.pth --weights-only
        """
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./base_models/"
        "2x_SuperUltraCompact_Pretrain_nf24_nc8_traiNNer.pth",
        help="Path to checkpoint file (default: %(default)s)",
    )
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help="Use weights_only=True when loading (safer, PyTorch 2.0+)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show progress bars (default: True)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    inspector = CheckpointInspector(
        args.checkpoint,
        weights_only=args.weights_only,
        verbose=not args.quiet,
    )
    success = inspector.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()