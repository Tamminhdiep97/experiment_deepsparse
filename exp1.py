from deepsparse import compile_model
from deepsparse.utils import generate_random_inputs
import onnxruntime as ort

from loguru import logger
import time
from tqdm import tqdm


onnx_filepath = "mobilenetv2-7.onnx"
batch_size = 1

# Generate random sample input
inputs = generate_random_inputs(onnx_filepath, batch_size)

# logger.info('Input type: {}'.format(type(inputs)))
# logger.info(type(inputs[0]))
# logger.info(inputs[0].shape)

# Compile and run
engine = compile_model(onnx_filepath, batch_size)
outputs = engine.run(inputs)

logger.info('benchmark deepsparse on {}'.format(onnx_filepath))
t_s = time.time()
for i in tqdm(range(1, 1001)):
    _ = engine.run(inputs)
t_e = time.time()
logger.info('Total time: {}s'.format(round(t_e-t_s, 5)))
logger.info('Avg time: {}s'.format(round((t_e-t_s)/1000, 5)))

sess_options = ort.SessionOptions()
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.inter_op_num_threads = 4

onnx_model = ort.InferenceSession(onnx_filepath, sess_options)

model_inputs = onnx_model.get_inputs()
input_names = [model_inputs[i].name for i in range(len(model_inputs))]
# warm up

for i in range(0, 10):
    _ = onnx_model.run(None, {input_names[0]: inputs[0]})

logger.info('benchmark onnx on {}'.format(onnx_filepath))

t_s = time.time()
for i in tqdm(range(1, 1001)):
    _ = onnx_model.run(None, {input_names[0]: inputs[0]})
t_e = time.time()
logger.info('Total time: {}s'.format(round(t_e-t_s, 5)))
logger.info('Avg time: {}s'.format(round((t_e-t_s)/1000, 5)))
