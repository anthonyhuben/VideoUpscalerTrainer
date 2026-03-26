# Export your trained model
python export_ncnn.py \
    --checkpoint ./checkpoints/best_model.pth \
    --output_dir ./ncnn_export \
    --input_size 1 3 540 960

# Output:
#   ./ncnn_export/model.param  (architecture)
#   ./ncnn_export/model.bin    (weights)
