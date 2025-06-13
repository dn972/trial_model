# Train
python main.py --config configs/default.yaml --mode train

# Inference
python main.py --config configs/default.yaml --mode infer

#Inference no gradient
CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES="1" python main.py --config configs/default_no_grad.yaml --mode inference_no_grad