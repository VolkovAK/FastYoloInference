import tensorrt as trt
import sys, os

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_file_path, max_batch, engine_file_path):
    """builds a new TensorRT engine and saves it."""
    """Takes an ONNX file and creates a TensorRT engine"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 28 # 256MiB
        builder.max_batch_size = 1 # does nothing

        # Parse model file
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        input_shape = network.get_input(0).shape[1:]

        config = builder.create_builder_config()
        profile = builder.create_optimization_profile()
        if max_batch > 1:
            profile.set_shape(
                    network.get_input(0).name, 
                    min=(1,) + input_shape, 
                    opt=(max_batch,) + input_shape, 
                    max=(max_batch,) + input_shape)
        else:
            const_shape = (1, ) + input_shape
            profile.set_shape(
                    network.get_input(0).name, 
                    min=const_shape, 
                    opt=const_shape, 
                    max=const_shape)

        if profile:  # validation check
            config.add_optimization_profile(profile)
        config.flags = 1 << int(trt.BuilderFlag.FP16) # try fp16, but not force it
        network.get_input(0).shape = (-1,) + input_shape # variable size batch

        engine = builder.build_engine(network, config)
        with open(engine_file_path, "wb") as f:
            f.write(engine.serialize())

