from __future__ import annotations

import argparse
import logging
from pathlib import Path

import tensorrt as trt

logger = logging.getLogger(__name__)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(
    onnx_path: Path,
    engine_path: Path,
    fp16: bool = True,
    min_shapes: str | None = None,
    opt_shapes: str | None = None,
    max_shapes: str | None = None,
) -> bool:
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error("ONNX parse error: %s", parser.get_error(i))
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    if min_shapes and opt_shapes and max_shapes:
        def _parse(spec: str) -> tuple[int, ...]:
            return tuple(int(x) for x in spec.split("x"))

        profile.set_shape(input_name, _parse(min_shapes), _parse(opt_shapes), _parse(max_shapes))
        config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("Engine build failed for %s", onnx_path)
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    logger.info("Wrote engine %s", engine_path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build TensorRT FP16 engines from ONNX")
    parser.add_argument("--onnx-dir", default="models/onnx")
    parser.add_argument("--engine-dir", default="models/engines")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--skip-espcn", action="store_true")
    parser.add_argument("--skip-flownet", action="store_true")
    args = parser.parse_args()

    onnx_dir = Path(args.onnx_dir)
    engine_dir = Path(args.engine_dir)
    results = []

    if not args.skip_espcn:
        espcn_onnx = onnx_dir / f"espcn_{args.scale}x.onnx"
        espcn_engine = engine_dir / f"espcn_{args.scale}x_fp16.engine"
        if espcn_onnx.is_file():
            ok = build_engine(
                espcn_onnx,
                espcn_engine,
                min_shapes="1x3x360x640",
                opt_shapes="1x3x1080x1920",
                max_shapes="1x3x1440x2560",
            )
            results.append(("ESPCN", ok))
        else:
            logger.warning("ESPCN ONNX missing: %s", espcn_onnx)
            results.append(("ESPCN", False))

    if not args.skip_flownet:
        flownet_onnx = onnx_dir / "flownet.onnx"
        flownet_engine = engine_dir / "flownet_fp16.engine"
        if flownet_onnx.is_file():
            ok = build_engine(
                flownet_onnx,
                flownet_engine,
                min_shapes="1x6x360x640",
                opt_shapes="1x6x1080x1920",
                max_shapes="1x6x1440x2560",
            )
            results.append(("FlowNet", ok))
        else:
            logger.warning("FlowNet ONNX missing: %s", flownet_onnx)
            results.append(("FlowNet", False))

    for name, ok in results:
        print(f"{name}: {'OK' if ok else 'FAILED'}")


if __name__ == "__main__":
    main()
